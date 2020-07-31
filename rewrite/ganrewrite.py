import copy
import os
import torch
import json
import random
import time
from utils import nethook, renormalize, pbar, tally, imgviz
from utils import show, labwidget, paintwidget
from collections import OrderedDict
import torchvision


(all_obs, all_weight, all_CinvK, all_kCinvK, e_val, e_vec, kbasis,
 row_dirs, q) = (None, None, None, None, None, None, None, None, None)

##########################################################################
# Algorithm
##########################################################################

# This base class is designed to rewrite a layer of a Progressive GAN.
# A subclass of this class is used for StyleGAN.


class ProgressiveGanRewriter(object):
    def __init__(self, model, zds, layernum,
                 cachedir=None,
                 low_rank_insert=True,  # Restrict insert to low rank context.
                 low_rank_gradient=False,  # Restrict gradient to low rank
                 use_linear_insert=False,  # Use linear insert
                 tight_paste=True,  # Paste to optimize over crop (vs whole image)
                 alpha_area=True,  # Target composites drawn area (vs boundrect)
                 key_method='zca'  # Other options: 'svd', 'gandissect'
                 ):
        self.firstlayer, self.lastlayer = self.maplayers(layernum)
        self.cachedir = cachedir
        self.tight_paste = tight_paste
        self.alpha_area = alpha_area
        self.key_method = key_method
        self.unit_rq = None
        self.unit_rs = None
        self.cad_rq = None
        self.low_rank_insert = low_rank_insert
        self.low_rank_gradient = low_rank_gradient
        self.use_linear_insert = use_linear_insert
        self.device = next(model.parameters()).device
        self.zds = zds
        self.model = copy.deepcopy(model)
        self.context_model = nethook.subsequence(
            self.model, upto_layer=self.firstlayer,
            share_weights=True)
        self.target_model = nethook.subsequence(
            self.model,
            first_layer=self.firstlayer,
            last_layer=self.lastlayer,
            share_weights=True)
        self.rendering_model = nethook.subsequence(
            self.model, after_layer=self.lastlayer,
            share_weights=True)
        with torch.no_grad():
            sample_z = self.get_z(0)
            sample_k = self.context_model(sample_z)
            sample_v = self.target_model(sample_k)
            sample_x = self.rendering_model(sample_v)
        self.k_shape = self.context_acts(sample_k).shape
        self.v_shape = self.target_acts(sample_v).shape
        self.x_shape = self.rendered_image(sample_x).shape
        self.c_matrix = self.collect_2nd_moment().to(self.device)
        self.zca_matrix = zca_from_cov(self.c_matrix)

    def model_state_dict(self):
        csd = self.context_model.state_dict()
        tsd = self.target_model.state_dict()
        rsd = self.rendering_model.state_dict()
        sd = {**csd, **tsd, **rsd}
        assert len(sd) == len(csd) + len(tsd) + len(rsd)
        return sd

    def maplayers(self, layernum):
        first = 'layer%d.conv' % layernum
        last = 'layer%d.conv' % layernum
        return first, last

    def collect_2nd_moment(self):
        '''
        Computes or loads cached uncentered covariance stats from r2m.npz.
        Returns the inverse of the matrix.
        '''
        with torch.no_grad(), pbar.quiet():  # quiet perhaps.
            def separate_key_reps(zbatch):
                acts = self.context_acts(
                    self.context_model(zbatch.to(self.device)))
                sep_pix = acts.permute(0, 2, 3, 1).reshape(-1, acts.shape[1])
                return sep_pix
            r2m = tally.tally_second_moment(separate_key_reps, self.zds,
                                            cachefile=self.rf('r2m.npz'))
            return r2m.moment()

    def covariance_adjusted_key(self, k, kout):
        return self.covariance_adjusted_query_key(k)

    def covariance_adjusted_query_key(self, k):
        # Returns C^{-1}k, computing it more stably than literally inverting C.
        if len(k.shape) == 1:
            return torch.lstsq(k[:, None], self.c_matrix)[0][:, 0]
        return torch.lstsq(k.permute(1, 0), self.c_matrix)[0].permute(1, 0)

    def zca_whitened_query_key(self, k):
        if len(k.shape) == 1:
            return torch.mm(self.zca_matrix, k[:, None])[:, 0]
        return torch.mm(self.zca_matrix, k.permute(1, 0)).permute(1, 0)

    def context_acts(self, context_out):
        return context_out

    def target_acts(self, target_out):
        return target_out

    def rendered_image(self, rendered_out):
        return rendered_out

    def sample_image_from_latent(self, z):
        return self.rendering_model(self.target_model(self.context_model(z)))

    def sample_image_patch(self, z, act_crop_size, seed=(None, None), act=False, size=4):
        feature_map = self.context_model(z)
        assert (act_crop_size <= feature_map.size(2))

        if seed[0] is not None:
            xi, yi = seed
        else:
            h, w = feature_map.shape[2:]
            xi = random.randint(0, h - act_crop_size)
            yi = random.randint(0, w - act_crop_size)

        xf, yf = xi + act_crop_size, yi + act_crop_size
        feature_map_cropped = feature_map[:, :, xi:xf, yi:yf]

        result = self.rendering_model(self.target_model(feature_map_cropped))

        if not act:
            return result
        else:
            highest_channel = feature_map_cropped.max(3)[0].max(2)[0].max(1)[1].item()
            iv = imgviz.ImageVisualizer((size, size))
            return result, iv.heatmap(feature_map_cropped[0, highest_channel], mode='nearest')

    def merge_target_output(self, target_out, new_acts, crop_bounds):
        '''
        Produce a renderable target output using new activations,
        getting other information from target_out as needed.  crop_bounds
        shows where in the original featuremap new acts was taken from.
        '''
        return new_acts

    def get_z(self, imgnum):
        return self.zds[imgnum][0][None].to(self.device)

    def apply_erase(self, request, rank=1, drank=30,
                    niter=2001, piter=10, lr=0.05, update_callback=None):
        '''Applies the editing specified in the JSON record, same
        format as saved by the UI.'''
        p_imgnum, p_mask = request['paste']
        key_examples = request.get('key', [(p_imgnum, p_mask)])
        goal_in, goal_out = self.erase_from_selection(
            p_imgnum, p_mask, key_examples, drank)
        mkey = self.multi_key_from_selection(key_examples, rank=rank)
        self.insert(goal_in, goal_out, mkey,
                    update_callback=update_callback,
                    niter=niter, piter=piter, lr=lr)

    def apply_edit(self, request, rank=1, niter=2001, piter=10, lr=0.05,
                   update_callback=None, single_key=-1):
        '''Applies the editing specified in the JSON record, same
        format as saved by the UI.

        single_key: if -1, will use all given keys as context.
                    Otherwise, will use key indexed by single_key
        '''
        o_imgnum, o_mask = request['object']
        p_imgnum, p_mask = request['paste']
        key_examples = request.get('key', [(p_imgnum, p_mask)])
        if single_key >= 0:
            print('Using only key', single_key, 'out of a total', len(key_examples))
            key_examples = [key_examples[single_key]]
        obj_acts, obj_output, obj_area, bounds = (
            self.object_from_selection(o_imgnum, o_mask))
        goal_in, goal_out, _, _ = self.paste_from_selection(
            p_imgnum, p_mask, obj_acts, obj_area)
        mkey = self.multi_key_from_selection(key_examples, rank=rank)
        return self.insert(goal_in, goal_out, mkey,
                           update_callback=update_callback,
                           niter=niter, piter=piter, lr=lr)

    def apply_overfit(self, request, niter=20001, lr=0.01,
                      update_callback=None):
        '''Applies the editing specified in the JSON record, same
        format as saved by the UI.'''
        o_imgnum, o_mask = request['object']
        p_imgnum, p_mask = request['paste']
        rgb_clip, _, obj_area, _ = self.rgb_from_selection(o_imgnum, o_mask)
        host_z, changed_rgb, bounds = self.rgbpaste_from_selection(
            p_imgnum, p_mask, rgb_clip, obj_area)
        self.all_weights_insert(changed_rgb, host_z, bounds=bounds,
                                update_callback=update_callback, niter=niter, lr=lr)

    def all_weights_insert(self, x, z, bounds=None, update_callback=None,
                           niter=20001, lr=0.01):

        obj_acts, obj_output, obj_area, bounds = (
            self.object_from_selection(o_imgnum, o_mask))
        goal_in, goal_out = self.paste_from_selection(
            p_imgnum, p_mask, obj_acts, obj_area)
        mkey = self.multi_key_from_selection(key_examples, rank=rank)
        self.insert(goal_in, goal_out, mkey,
                    update_callback=update_callback,
                    niter=niter, piter=piter, lr=lr)

    def detach(self, v):
        return v.detach()

    def target_weights(self):
        return [p for n, p in self.target_model.named_parameters()
                if 'weight' in n][0]

    def zero(self, context, amount=0.0):
        weight = self.target_weights()
        with torch.no_grad():
            ortho_weight = weight - projected_conv(weight, context)
            weight[...] = ortho_weight + (
                amount * projected_conv(torch.ones_like(weight), context))

        def compute_loss():
            return torch.nn.functional.l1_loss(self.target_acts(val),
                                               self.target_acts(self.target_model(key)))

    def linear_insert(self, key, val, context=None, update_callback=None,
                      niter=2001, lr=0.05, return_timing=False):
        if return_timing:
            torch.cuda.synchronize()
            st = time.time()
        nethook.set_requires_grad(False, self.model)
        key, val = [self.detach(d) for d in [key, val]]
        original_weight = self.target_weights()
        hooked_module = [module for module in self.target_model.modules()
                         if getattr(module, 'weight', None) is original_weight][0]
        del hooked_module._parameters['weight']
        ws = original_weight.shape
        lambda_param = torch.zeros(
            ws[0], ws[1], context.shape[0],
            ws[3], ws[4], device=original_weight.device,
            requires_grad=True)
        old_forward = hooked_module.forward

        def new_forward(x):
            # weight_1 = weight_0 + Lambda D
            hooked_module.weight = (original_weight + torch.einsum('godyx, di -> goiyx', lambda_param, context))
            result = old_forward(x)
            return result
        hooked_module.forward = new_forward

        # when computing the loss, hook the weights to be modified by Lambda D
        def compute_loss():
            loss = torch.nn.functional.l1_loss(self.target_acts(val),
                                               self.target_acts(self.target_model(key)))
            return loss

        # run the optimizer
        params = [lambda_param]
        optimizer = torch.optim.Adam(params, lr=lr)
        for it in range(niter):
            with torch.enable_grad():
                loss = compute_loss()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if update_callback is not None:
                    update_callback(it, loss)
        with torch.no_grad():
            # OK now fill in the learned weights and undo the hook.
            original_weight[...] = (original_weight + torch.einsum('godyx, di -> goiyx', lambda_param, context))
            del hooked_module.weight
            hooked_module.register_parameter('weight', original_weight)
            hooked_module.forward = old_forward
        if return_timing:
            torch.cuda.synchronize()
            et = time.time()
            return (et - st) * 1000

    def insert(self, key, val, context=None, update_callback=None,
               niter=2001, piter=10, lr=0.05, return_timing=False):
        if self.use_linear_insert:
            return self.linear_insert(key, val, context,
                                      update_callback=update_callback,
                                      niter=niter, lr=lr,
                                      return_timing=return_timing)
        if return_timing:
            torch.cuda.synchronize()
            st = time.time()
        # print('inserting keys %s' % list(key.keys()))
        key, val = [self.detach(d) for d in [key, val]]

        def compute_loss():
            return torch.nn.functional.l1_loss(self.target_acts(val),
                                               self.target_acts(self.target_model(key)))
        # set up optimizer
        weight = self.target_weights()
        params = [weight]
        if self.low_rank_insert or self.low_rank_gradient:
            # The assumption now is that context is orthonormal.
            with torch.no_grad():
                ortho_weight = weight - projected_conv(weight, context)
        optimizer = torch.optim.Adam(params, lr=lr)

        for it in range(niter):
            with torch.enable_grad():
                loss = compute_loss()
                optimizer.zero_grad()
                loss.backward()
                # If we want to project the gradient, we can do this:
                if self.low_rank_gradient:
                    weight.grad[...] = projected_conv(weight.grad, context)
                optimizer.step()
                if update_callback is not None:
                    update_callback(it, loss)
                # Project to rank-one over context direction
                if self.low_rank_insert and (it % piter == 0 or it == niter - 1):
                    with torch.no_grad():
                        weight[...] = (
                            ortho_weight + projected_conv(weight, context))
        if return_timing:
            torch.cuda.synchronize()
            et = time.time()
            return (et - st) * 1000

    def all_weights_insert(self, x, z, bounds=None, update_callback=None,
                           niter=20001, lr=0.01):
        x, z = [self.detach(d) for d in [x, z]]
        vgg = torchvision.models.vgg16(pretrained=True)
        VF = nethook.subsequence(vgg.features, last_layer='20').to(x.device)

        def compute_loss():
            out = self.model(z)
            if bounds is None:
                gt, pred = x, out
            else:
                t, l, b, r = bounds
                gt, pred = [d[:, :, t:b, l:r] for d in [x, out]]
            # Regularizing like this doesn't really help.
            # reg = 1e+2 * sum(torch.nn.functional.l1_loss(
            #   orig_p, cur_p) for (orig_p, cur_p) in zip(orig_params, params))
            return torch.nn.functional.l1_loss(gt, pred) + (
                1e-2 * torch.nn.functional.mse_loss(VF(gt), VF(pred)))
        # set up optimizer
        nethook.set_requires_grad(False, self.model)
        params = list(self.model.parameters())
        nethook.set_requires_grad(True, *params)
        optimizer = torch.optim.Adam(params, lr=lr)

        for it in range(niter):
            with torch.enable_grad():
                loss = compute_loss()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if update_callback is not None:
                    update_callback(it, loss)

    def multi_key_from_selection(self, imgnum_mask_pairs, rank=1,
                                 key_method=None):
        global all_obs, all_weight, all_CinvK, all_kCinvK, e_val, e_vec, kbasis, row_dirs, q
        if key_method is None:
            key_method = self.key_method
        with torch.no_grad():
            if key_method in ['zca']:
                accumulated_obs = []
                for imgnum, mask in imgnum_mask_pairs:
                    k_outs = self.context_model(self.get_z(imgnum))
                    k_acts = self.context_acts(k_outs)
                    area = renormalize.from_url(mask, target='pt',
                                                size=self.k_shape[2:])[0]
                    accumulated_obs.append((
                        k_acts.permute(0, 2, 3, 1).reshape(-1, k_acts.shape[1]),
                        k_outs,
                        area.view(-1)[:, None].to(k_acts.device)))
                all_obs = torch.cat([obs[(w > 0).nonzero()[:, 0], :]
                                     for obs, _, w in accumulated_obs])
                all_weight = torch.cat([w[w > 0]
                                        for _, _, w in accumulated_obs])
                all_zca_k = torch.cat([
                    (w * self.zca_whitened_query_key(obs)
                     )[(w > 0).nonzero()[:, 0], :]
                    for obs, outs, w in accumulated_obs])
                # all_zca_k is already transposed
                _, _, q = all_zca_k.svd(compute_uv=True)
                # Get the top rank e_vecs in whitened space
                top_e_vec = q[:, :rank]
                # Transform them into rowspace. (Same as multiplying
                # by ZCA matrix a 2nd time.)
                row_dirs = self.zca_whitened_query_key(top_e_vec.t())
                just_avg = (all_zca_k).sum(0)
                # Orthogonalize row_dirs
                q, r = torch.qr(row_dirs.permute(1, 0))
                # Flip the first eigenvec to agree with avg direction.
                signs = (q * just_avg[:, None]).sum(0).sign()
                q = q * signs[None, :]
                return q.permute(1, 0)
            if key_method == 'gandissect':
                # Unit-wise-keys select D using a gandissect-like rule.
                # We score a unit by how unusual the visible values are.
                # i.e., log probability = weighted-sum log probabilities
                # Here we use explicitly counted quantiles to estimate probs.
                accumulated_obs = []
                for imgnum, mask in imgnum_mask_pairs:
                    k_outs = self.context_model(self.get_z(imgnum))
                    k_acts = self.context_acts(k_outs)
                    area = renormalize.from_url(mask, target='pt',
                                                size=self.k_shape[2:])[0]
                    accumulated_obs.append((
                        k_acts.permute(0, 2, 3, 1).reshape(-1, k_acts.shape[1]),
                        area.view(-1)[:, None].to(k_acts.device)))
                all_obs = torch.cat([obs for obs, _ in accumulated_obs])
                all_weight = torch.cat([w for _, w in accumulated_obs])
                rq = self.quantiles_for_units()
                all_logscore = -torch.log(
                    1.0 - rq.normalize(all_obs.permute(1, 0))).permute(1, 0)
                mean_logscore = ((all_logscore * all_weight).sum(0) / sum(all_weight))
                top_coords = mean_logscore.sort(descending=True)[1][:rank]
                result = torch.zeros(rank, all_obs.shape[1],
                                     device=all_obs.device)
                result[torch.arange(rank), top_coords] = 1.0
                # print('top_coords', top_coords.tolist())
                return result
            # Old SVD method
            assert key_method in ['svd', 'mean']
            accumulated_k = []
            for imgnum, mask in imgnum_mask_pairs:
                k_outs = self.context_model(self.get_z(imgnum))
                k_acts = self.context_acts(k_outs)
                area = renormalize.from_url(mask, target='pt',
                                            size=self.k_shape[2:])[0]
                weighted_k = (k_acts[0] * area[None].to(self.device)
                              ).permute(1, 2, 0).view(-1, k_acts.shape[1])
                nonzero_k = weighted_k[weighted_k.norm(2, dim=1) > 0]
                accumulated_k.append((nonzero_k, k_outs))
            all_k = torch.cat([self.covariance_adjusted_key(nonzero_k, k_outs)
                               for nonzero_k, k_outs in accumulated_k])
            just_avg = all_k.mean(0)
            if key_method == 'mean':
                assert rank == 1
                return just_avg[None, :] / just_avg.norm()
            u, s, v = torch.svd(all_k.permute(1, 0), some=False)
            if (just_avg * u[:, 0]).sum() < 0:
                # Flip the first singular vectors to agree with avg direction
                u[:, 0] = -u[:, 0]
                v[:, 0] = -v[:, 0]
            assert u.shape[1] >= rank
            return u.permute(1, 0)[:rank]

    def query_key_from_selection(self, imgnum, mask):
        area = renormalize.from_url(mask, target='pt', size=self.k_shape[2:])[0]
        with torch.no_grad():
            k_outs = self.context_model(self.get_z(imgnum))
            k_acts = self.context_acts(k_outs)
            mean = (k_acts[0] * area[None].to(self.device)
                    ).sum(2).sum(1) / (1e-10 + area.sum())
        # Old version: zeroed all but top 100 components of k; and
        # also omit channel 418!!  Here we just omit 418.
        # TODO: does it work without the blacklist?
        # blacklist = [418]
        # k[blacklist] = 0
        k = self.covariance_adjusted_query_key(mean)
        k = k / (1e-10 + k.norm(2))
        return k

    def is_empty_mask(self, mask):
        area = renormalize.from_url(mask, target='pt')[0]
        return area.sum() == 0.0

    def object_from_selection(self, imgnum, mask):
        area = renormalize.from_url(mask, target='pt', size=self.v_shape[2:])[0]
        with torch.no_grad():
            k_output = self.context_model(self.get_z(imgnum))
            v_output = self.target_model(k_output)
            v_acts = self.target_acts(v_output)
        t, l, b, r = positive_bounding_box(area)
        obj_activations = v_acts[:, :, t:b, l:r]
        obj_area = area[t:b, l:r]
        return obj_activations, v_output, obj_area, (t, l, b, r)

    def normdissect_units(self, imgnum_mask_pairs, rank):
        with torch.no_grad():
            accumulated_obs = []
            for imgnum, mask in imgnum_mask_pairs:
                k_outs = self.context_model(self.get_z(imgnum))
                k_acts = self.context_acts(k_outs)
                area = renormalize.from_url(mask, target='pt',
                                            size=self.k_shape[2:])[0]
                accumulated_obs.append((
                    k_acts.permute(0, 2, 3, 1).reshape(-1, k_acts.shape[1]),
                    area.view(-1)[:, None].to(k_acts.device)))
            all_obs = torch.cat([obs for obs, _ in accumulated_obs])
            all_weight = torch.cat([w for _, w in accumulated_obs])
            square_scale = self.square_scales_for_units().to(all_obs.device)
            all_logscore = all_obs.pow(2) / square_scale[None, :]
            mean_logscore = ((all_logscore * all_weight).sum(0) / sum(all_weight))
            top_coords = mean_logscore.sort(descending=True)[1][:rank]
            return top_coords

    def erase_from_selection(self, imgnum, mask, context_mask_pairs, rank):
        k_area = renormalize.from_url(
            mask, target='pt', size=self.k_shape[2:])[0]
        area = renormalize.from_url(
            mask, target='pt', size=self.v_shape[2:])[0]
        source_outputs = self.context_model(self.get_z(imgnum))
        source_acts = self.context_acts(source_outputs)
        unchanged_outputs = self.target_model(source_outputs)
        source_acts_without_units = source_acts.clone()
        d_units = self.normdissect_units(context_mask_pairs, rank)
        source_acts_without_units[:, d_units] = 0.0
        d_erased_in = self.merge_target_output(
            source_outputs, source_acts_without_units, None)
        d_erased_out = self.target_model(d_erased_in)
        target_acts = self.target_acts(d_erased_out)
        if self.tight_paste:
            source_bounds = positive_bounding_box(k_area)
            target_bounds = positive_bounding_box(area)
        else:
            source_bounds, target_bounds = None, None
        goal_in = self.merge_target_output(
            source_outputs, source_acts, source_bounds)
        goal_out = self.merge_target_output(
            unchanged_outputs, target_acts, target_bounds)
        return goal_in, goal_out

    def paste_from_selection(self, imgnum, mask, obj_acts, obj_area):
        area = renormalize.from_url(
            mask, target='pt', size=self.v_shape[2:])[0]
        source_outputs = self.context_model(self.get_z(imgnum))
        source_acts = self.context_acts(source_outputs)
        unchanged_outputs = self.target_model(source_outputs)
        unchanged_acts = self.target_acts(unchanged_outputs)
        target_acts, bounds = paste_clip_at_center(
            unchanged_acts, obj_acts, centered_location(area),
            obj_area if self.alpha_area else None)
        full_target_acts = target_acts
        if self.tight_paste:
            source_acts, target_acts, source_bounds, target_bounds = (
                crop_clip_to_bounds(source_acts, target_acts, bounds))
        else:
            source_bounds, target_bounds = None, None
        goal_in = self.merge_target_output(
            source_outputs, source_acts, source_bounds)
        goal_out = self.merge_target_output(
            unchanged_outputs, target_acts, target_bounds)
        viz_out = self.merge_target_output(
            unchanged_outputs, full_target_acts, None)
        return goal_in, goal_out, viz_out, bounds

    def rgb_from_selection(self, imgnum, mask):
        area = renormalize.from_url(mask, target='pt', size=self.x_shape[2:])[0]
        with torch.no_grad():
            x_output = self.model(self.get_z(imgnum))
        t, l, b, r = positive_bounding_box(area)
        rgb_clip = x_output[:, :, t:b, l:r]
        obj_area = area[t:b, l:r]
        return rgb_clip, x_output, obj_area, (t, l, b, r)

    def rgbpaste_from_selection(self, imgnum, mask, obj_rgb, obj_area):
        with torch.no_grad():
            area = renormalize.from_url(
                mask, target='pt', size=self.x_shape[2:])[0]
            source_z = self.get_z(imgnum)
            unchanged_rgb = self.model(source_z)
            changed_rgb, bounds = paste_clip_at_center(
                unchanged_rgb, obj_rgb, centered_location(area), obj_area)
        return source_z, changed_rgb, bounds

    def square_scales_for_units(self):
        if self.unit_rs is None:
            with pbar.quiet(), torch.no_grad():
                def squared_unit_values(zbatch):
                    acts = self.context_acts(self.context_model(
                        zbatch.to(self.device))).detach()
                    flattened = acts.permute(0, 2, 3, 1).reshape(-1, acts.shape[1])
                    return flattened.pow(2)
                self.unit_rs = tally.tally_mean(
                    squared_unit_values, self.zds,
                    cachefile=self.rf('unit_rs.npz')).mean()
        return self.unit_rs

    def quantiles_for_units(self):
        if self.unit_rq is None:
            with pbar.quiet(), torch.no_grad():
                def flattened_unit_values(zbatch):
                    acts = self.context_acts(self.context_model(
                        zbatch.to(self.device))).detach()
                    flattened = acts.permute(0, 2, 3, 1).reshape(-1, acts.shape[1])
                    return flattened
                self.unit_rq = tally.tally_quantile(
                    flattened_unit_values, self.zds,
                    cachefile=self.rf('unit_rq.npz'))
        return self.unit_rq

    def quantiles_for_covariance_adjusted_directions(self):
        if self.cad_rq is None:
            with pbar.quiet(), torch.no_grad():
                def flattened_unit_values(zbatch):
                    outs = self.context_model(zbatch.to(self.device))
                    acts = self.context_acts(outs)
                    flattened = acts.permute(0, 2, 3, 1).reshape(-1, acts.shape[1])
                    # TODO: support unrolling batch outs.
                    adjusted = self.covariance_adjusted_key(flattened, outs)
                    return adjusted
                self.cad_rq = tally.tally_quantile(
                    flattened_unit_values, self.zds,
                    cachefile=self.rf('unit_cad.npz'))
        return self.cad_rq

    def ranking_for_key(self, key, k=12):
        tensorkey = key.to(self.device)[None, :, None, None]
        with pbar.quiet(), torch.no_grad():
            def image_max_sel(zbatch):
                acts = self.context_acts(self.context_model(
                    zbatch.to(self.device)))
                heatmap = (acts * tensorkey).sum(dim=1)
                maxmap = heatmap.view(heatmap.shape[0], -1).max(1)[0]
                flatmap = heatmap.view(-1)[:, None]
                return maxmap, flatmap
            topk, rq = tally.tally_topk_and_quantile(
                image_max_sel, self.zds, k=k)
        return topk.result()[1], rq

    def render_object(self, target_output, obj_area=None, box=None):
        with torch.no_grad():
            imgdata = self.rendered_image(
                self.rendering_model(target_output))
        if box is None:
            return renormalize.as_image(imgdata[0])
        # Make a mask from the box
        t, l, b, r = box
        lowres = torch.zeros(self.v_shape[2:])
        lowres[t:b, l:r] = 1
        iv = imgviz.ImageVisualizer(imgdata.shape[2:])
        return iv.masked_image(imgdata, activations=lowres, level=0.0,
                               border_color=[255, 0, 0], thickness=3)

    def render_image(self, imgnum, key=None, level=None, mask=None, **kwargs):
        with torch.no_grad():
            context_output = self.context_model(self.get_z(imgnum))
            target_output = self.target_model(context_output)
            imgdata = self.rendered_image(self.rendering_model(target_output))
        if key is not None and level is not None:
            tensorkey = key.to(self.device)[None, :, None, None]
            acts = self.context_acts(self.context_model(self.get_z(imgnum)))
            heatmap = (acts[...] * tensorkey).sum(dim=1)[0]
            iv = imgviz.ImageVisualizer(imgdata.shape[2:])
            return iv.masked_image(imgdata, heatmap, level=level, **kwargs)
        elif mask is not None:
            iv = imgviz.ImageVisualizer(imgdata.shape[2:])
            return iv.masked_image(imgdata, mask=mask, **kwargs)
        return renormalize.as_image(imgdata[0])

    def render_image_batch(self, imgnums, key=None, level=None, **kwargs):
        batch_size = 3
        results = []
        for i in range(0, len(imgnums), batch_size):
            imgnum_batch = imgnums[i:i + batch_size]
            with torch.no_grad():
                z_batch = torch.cat([self.get_z(imgnum)
                                     for imgnum in imgnum_batch])
                context_output = self.context_model(z_batch)
                target_output = self.target_model(context_output)
                imgdata_batch = self.rendered_image(
                    self.rendering_model(target_output))
            if key is not None and level is not None:
                tensorkey = key.to(self.device)[None, :, None, None]
                acts = self.context_acts(self.context_model(z_batch))
                heatmap = (acts[...] * tensorkey).sum(dim=1)
                iv = imgviz.ImageVisualizer(imgdata_batch.shape[2:])
                results.extend(
                    [iv.masked_image(imgdata, heatmap[j], level=level,
                                     **kwargs)
                     for j, imgdata in enumerate(imgdata_batch)])
            else:
                results.extend([
                    renormalize.as_image(imgdata) for imgdata in imgdata_batch])
        return results

    def rf(self, fn):
        if self.cachedir is None:
            return None
        return os.path.join(self.cachedir, fn)


class SeqStyleGanRewriter(ProgressiveGanRewriter):
    def __init__(self, model, zds, layernum, **kwargs):
        super().__init__(model, zds, layernum, **kwargs)

    def maplayers(self, layernum):
        first = 'layer%d.sconv.mconv.dconv' % layernum
        last = 'layer%d.sconv.activate' % layernum
        return first, last

    def sample_image_patch(self, z, act_crop_size, seed=(None, None), act=False, size=None):
        out = self.context_model(z)
        feature_map = out['fmap']
        img = out['output']
        assert (act_crop_size <= feature_map.size(2))

        if seed[0] is not None:
            xi, yi = seed
        else:
            h, w = feature_map.shape[2:]
            xi = random.randint(0, h - act_crop_size)
            yi = random.randint(0, w - act_crop_size)

        xf, yf = xi + act_crop_size, yi + act_crop_size
        feature_map_cropped = feature_map[:, :, xi:xf, yi:yf]

        if feature_map.shape[2:] == img.shape[2:]:
            img_cropped = img[:, :, xi:xf, yi:yf]
        else:
            # stylegan's running image is bigger than our activation, so we need to upsample the mask.
            img_cropped = img[:, :, 2 * xi:2 * xf, 2 * yi:2 * yf]

        out['output'] = img_cropped
        out['fmap'] = feature_map_cropped
        result = self.rendering_model(self.target_model(out))

        if not act:
            return result
        else:
            highest_channel = feature_map_cropped.max(3)[0].max(2)[0].max(1)[1].item()
            iv = imgviz.ImageVisualizer((size, size))
            return result, iv.heatmap(feature_map_cropped[0, highest_channel], mode='nearest')

    def covariance_adjusted_query_key(self, k):
        if len(k.shape) == 1:
            return torch.lstsq(k[:, None], self.c_matrix)[0][:, 0]
        return torch.lstsq(k.permute(1, 0), self.c_matrix)[0].permute(1, 0)

    def covariance_adjusted_key(self, k, kout):
        return self.covariance_adjusted_query_key(k)

    def detach(self, v):
        if isinstance(v, dict):
            return type(v)({k: d.detach() for k, d in v.items()})
        return v.detach()

    def context_acts(self, context_out):
        return context_out.fmap

    def target_acts(self, target_out):
        return target_out.fmap

    def rendered_image(self, rendered_out):
        return rendered_out

    def merge_target_output(self, target_out, new_acts, crop_bounds):
        newcopy = type(target_out)(
            {k: d.detach() for k, d in target_out.items()})
        if crop_bounds is not None:
            t, l, b, r = crop_bounds
            newcopy.output = newcopy.output[:, :, t:b, l:r]
        newcopy.fmap = new_acts
        return newcopy


class SeqTinyStyleGanRewriter(SeqStyleGanRewriter):
    def __init__(self, model, zds, layernum, **kwargs):
        super().__init__(model, zds, layernum, **kwargs)

    def maplayers(self, layernum):
        first = 'layer%d.sconv.mconv.dconv' % layernum
        last = 'layer%d.sconv.mconv.dconv' % layernum
        return first, last


class SeqPreStyleGanRewriter(SeqStyleGanRewriter):
    def __init__(self, model, zds, layernum, **kwargs):
        super().__init__(model, zds, layernum, **kwargs)

    def maplayers(self, layernum):
        first = 'layer%d.sconv.mconv.adain' % layernum
        last = 'layer%d.sconv.activate' % layernum
        return first, last

    def covariance_adjusted_key(self, k, kout):
        assert 'adain' in self.firstlayer
        # The idea here is that we should be constrained to the low
        # rank subspace corresponding to (CS)^{-1} k.
        assert kout.style.shape[0] == 1
        # cs = self.c_matrix * kout.style[0][:,None]  # SC version
        cs = self.c_matrix * kout.style[0][None, :]  # CS version, correct I think
        if len(k.shape) == 1:
            return torch.lstsq(k[:, None], cs)[0][:, 0]
        return torch.lstsq(k.permute(1, 0), cs)[0].permute(1, 0)


##########################################################################
# UI
##########################################################################

class GanRewriteWidget(labwidget.Widget):

    def __init__(self, gw, mask_dir=None, size=256, num_canvases=9):
        super().__init__()
        self.gw = gw
        self.size = size
        self.savedir = self.gw.cachedir if mask_dir is None else mask_dir
        self.original_model = copy.deepcopy(gw.model)
        self.request = {}
        self.imgnum_textbox = labwidget.Textbox('0-%d' % (num_canvases - 1)
                                                ).on('value', self.change_numbers)
        self.msg_out = labwidget.Div()
        self.loss_out = labwidget.Div()
        self.query_out = labwidget.Div()
        self.object_out = labwidget.Div()
        self.target_out = labwidget.Div()
        self.keytray_div = labwidget.Div(style={'display': 'none'})
        self.keytray_menu = labwidget.Menu(
        ).on('selection', self.repaint_key_tray)
        self.keytray_removebtn = labwidget.Button('remove'
                                                  ).on('click', self.keytray_remove)
        self.keytray_showbtn = labwidget.Button('show'
                                                ).on('click', self.keytray_show)
        self.keytray_querybtn = labwidget.Button('query'
                                                 ).on('click', self.keytray_query)
        self.keytray_zerobtn = labwidget.Button('zero'
                                                ).on('click', self.keytray_zero)
        self.keytray_canvas = paintwidget.PaintWidget(
            width=self.size, height=self.size,
            vanishing=False, disabled=True)
        self.keytray_div.show([
            [
                self.keytray_menu,
                self.keytray_removebtn,
                self.keytray_showbtn,
                self.keytray_zerobtn,
                self.keytray_querybtn,
            ],
            [self.keytray_canvas]])
        self.key_btn = labwidget.Button('key').on('click', self.key_add)
        self.query_btn = labwidget.Button('query').on('click', self.query)
        self.highlight_btn = labwidget.Button('highlight'
                                              ).on('click', self.toggle_highlight)
        self.object_btn = labwidget.Button('object'
                                           ).on('click', self.pick_object)
        self.snap_btn = labwidget.Button('snap'
                                         ).on('click', self.snapshot_images)
        self.brushsize_textbox = labwidget.Textbox(10, desc='brush: ', size=3
                                                   ).on('value', self.change_brushsize)
        self.rank_textbox = labwidget.Textbox(
            '1', desc='rank: ', size=4)
        self.paste_niter_textbox = labwidget.Textbox(
            '2001', desc='paste niter: ', size=8)
        self.paste_piter_textbox = labwidget.Textbox(
            '10', desc='proj every: ', size=4)
        self.paste_lr_textbox = labwidget.Textbox(
            '0.05', desc='paste lr: ', size=8)
        self.paste_btn = labwidget.Button('paste').on('click', self.paste)
        self.erase_btn = labwidget.Button('erase').on('click', self.exec_erase)
        self.exec_btn = labwidget.Button('exec').on('click', self.exec_request)
        self.overfit_btn = labwidget.Button('overfit').on(
            'click', self.exec_overfit)
        self.revert_btn = labwidget.Button('revert').on('click', self.revert)
        self.saved_list = labwidget.Datalist(choices=self.saved_names())
        self.load_btn = labwidget.Button('load').on('click', self.tryload)
        self.save_btn = labwidget.Button('save').on('click', self.save)
        self.sel = list(range(num_canvases))
        self.overwriting = True
        self.obj_acts = None
        self.query_key = None
        self.query_vis = False
        self.query_rq = None
        self.clipped_activations = None
        self.canvas_array = []
        self.snap_image_array = []
        for i in range(num_canvases):
            self.canvas_array.append(paintwidget.PaintWidget(
                image=renormalize.as_url(
                    self.gw.render_image(i)),
                width=self.size, height=self.size
            ).on('mask', self.change_mask))
            self.canvas_array[-1].index = i
            self.snap_image_array.append(
                labwidget.Image(style={'margin-top': 0,
                                       'max-width': '%dpx' % self.size,
                                       'max-height': '%dpx' % self.size}))
            self.snap_image_array[-1].index = i
        self.current_mask_item = None

    def change_brushsize(self):
        brushsize = int(self.brushsize_textbox.value)
        for c in self.canvas_array:
            c.brushsize = brushsize

    def repaint_key_tray(self):
        if 'key' not in self.request or len(self.request['key']) == 0:
            self.keytray_div.style = {'display': 'none'}
            return
        keymasks = OrderedDict([
            (imgnum, mask) for imgnum, mask in self.request['key']])
        self.keytray_menu.choices = list(keymasks.keys())
        self.keytray_div.style = {'display': 'block'}
        if self.keytray_menu.selection is None or (
                int(self.keytray_menu.selection) not in keymasks):
            sel = int(self.keytray_menu.choices[-1])
            self.keytray_menu.selection = sel
        else:
            sel = int(self.keytray_menu.selection)
        self.keytray_canvas.image = renormalize.as_url(
            self.gw.render_image(sel, None))
        self.keytray_canvas.mask = keymasks[sel]

    def keytray_remove(self):
        if self.keytray_menu.selection is None:
            return
        if 'key' not in self.request or len(self.request['key']) == 0:
            return
        sel = int(self.keytray_menu.selection)
        if sel is None:
            return True
        index = [imgnum for imgnum, _ in self.request['key']].index(sel)
        if index >= 0:
            self.request['key'].pop(index)
            if len(self.request['key']) == 0:
                del self.request['key']
            self.keytray_menu.selection = None
            self.repaint_key_tray()

    def keytray_show(self):
        if 'key' not in self.request or len(self.request['key']) == 0:
            return
        self.sel = [i for i, _ in self.request['key']]
        self.imgnum_textbox.value = ','.join(str(i) for i in self.sel)
        self.repaint_canvas_array()

    def keytray_query(self):
        if 'key' not in self.request or len(self.request['key']) == 0:
            return
        mkey = self.gw.multi_key_from_selection(self.request['key'], rank=1)
        self.exec_query(key=mkey[0])

    def keytray_zero(self):
        if 'key' not in self.request or len(self.request['key']) == 0:
            return
        rank = int(self.rank_textbox.value)
        mkey = self.gw.multi_key_from_selection(self.request['key'], rank=rank)
        self.gw.zero(mkey)
        self.repaint_canvas_array()
        self.show_msg(f'zeroed key value from model')

    def key_add(self):
        if self.current_mask_item is None or not self.canvas_array[self.current_mask_item].mask:
            return
        imgnum = self.sel[self.current_mask_item]
        mask = self.canvas_array[self.current_mask_item].mask
        if 'key' in self.request:
            keymasks = OrderedDict([
                (imgnum, mask) for imgnum, mask in self.request['key']])
        else:
            keymasks = {}
        keymasks[imgnum] = mask
        self.request['key'] = list(keymasks.items())
        self.keytray_menu.selection = imgnum
        self.repaint_key_tray()

    def repaint_canvas_array(self):
        level = (self.query_rq.quantiles(0.999)[0]
                 if (self.query_vis and self.query_rq) else None)
        images = self.gw.render_image_batch(self.sel,
                                            self.query_key if self.query_vis else None, level)
        for i, canvas in enumerate(self.canvas_array):
            canvas.mask = ''
            size = (canvas.height, canvas.width) if canvas.height else None
            if i < len(self.sel):
                canvas.image = renormalize.as_url(images[i], size=size)
            else:
                canvas.image = ''

    def snapshot_images(self):
        for canvas, snap in zip(self.canvas_array, self.snap_image_array):
            snap.src = canvas.image

    def clear_images(self):
        for snap in self.snap_image_array:
            snap.src = ''

    def change_numbers(self, c):
        sel = []
        for p in [r.split('-') for r in self.imgnum_textbox.value.split(',')]:
            try:
                bottom = int(p[0])
                top = int(p[1]) if len(p) > 1 else bottom
                sel.extend([i for i in range(bottom, top + 1)
                            if 0 <= i < len(self.gw.zds)])
            except Exception as e:
                print('exception {}'.format(e))
                pass
            if len(sel) >= len(self.canvas_array):
                sel = sel[:len(self.canvas_array)]
        self.sel = sel
        self.repaint_canvas_array()
        self.clear_images()

    def change_mask(self, ev):
        i = ev.target.index
        for j in range(len(self.canvas_array)):
            if j != i:
                self.canvas_array[j].mask = ''
        self.current_mask_item = i

    def set_mask(self, imgnum, mask):
        assert imgnum in self.sel
        for i in range(len(self.canvas_array)):
            if self.sel[i] != imgnum:
                self.canvas_array[i].mask = ''
            else:
                self.canvas_array[i].mask = mask
                self.current_mask_item = i

    def toggle_highlight(self):
        self.query_vis = not self.query_vis
        self.repaint_canvas_array()

    def query(self):
        if self.overwriting or 'query' not in self.request:
            if self.current_mask_item is None or not self.canvas_array[self.current_mask_item].mask:
                return
            imgnum = self.sel[self.current_mask_item]
            mask = self.canvas_array[self.current_mask_item].mask
            self.request['query'] = (imgnum, mask)
        self.exec_query()

    def exec_query(self, key=None, no_repaint=False):
        if key is None:
            imgnum, mask = self.request['query']
            key = self.gw.query_key_from_selection(imgnum, mask)
        sel, rq = self.gw.ranking_for_key(key, k=len(self.canvas_array))
        self.sel = sel.tolist()
        self.query_rq = rq
        self.imgnum_textbox.value = ','.join(str(i) for i in self.sel)
        self.query_key = key
        self.query_key_units = key.sort(descending=True)[1][:10].tolist()
        self.query_out.clear()
        self.query_out.print('query: ', self.query_key_units)
        if not no_repaint:
            self.repaint_canvas_array()
            self.clear_images()
        self.show_msg('query done')

    def pick_object(self):
        if self.overwriting or 'object' not in self.request:
            if self.current_mask_item is None or not self.canvas_array[self.current_mask_item].mask:
                return
            imgnum = self.sel[self.current_mask_item]
            mask = self.canvas_array[self.current_mask_item].mask
            self.request['object'] = (imgnum, mask)
        self.exec_object()

    def exec_object(self, obj_acts=None, obj_area=None, obj_output=None,
                    bounds=None):
        if obj_acts is None:
            imgnum, mask = self.request['object']
            obj_acts, obj_output, obj_area, bounds = (
                self.gw.object_from_selection(imgnum, mask))
        if obj_area is None:
            obj_area = torch.ones(obj_acts.shape[2:], device=obj_acts.device)
        self.obj_acts, self.obj_area = obj_acts, obj_area
        highest_channel = obj_acts.max(3)[0].max(2)[0].max(1)[1].item()
        cropped_out = self.gw.merge_target_output(obj_output, obj_acts, bounds)
        imgout = self.gw.render_object(cropped_out, obj_area)
        iv = imgviz.ImageVisualizer((imgout.height, imgout.width))
        self.object_out.show(
            ['value (from %d):' % imgnum, [imgout],
             [iv.heatmap(obj_acts[0, highest_channel], mode='nearest')]])
        self.show_msg('picked object')

    def show_request_mask(self, field='object', index=None, **kwargs):
        # For generating high-resolution figures: directly visualize a mask.
        if field not in self.request:
            print(f'No {field} selected')
            return
        if field == 'key':
            if index >= len(self.request[field]):
                print(f'No {index}th entry in key')
                return
            imgnum, mask = self.request[field][index]
        else:
            imgnum, mask = self.request[field]
        area = (renormalize.from_url(mask, target='pt',
                                     size=self.gw.x_shape[2:])[0] > 0.25)
        imgout = self.gw.render_image(imgnum, mask=area, **kwargs)
        show(imgout)

    def revert(self):
        with torch.no_grad():
            self.gw.model.load_state_dict(self.original_model.state_dict())
        self.repaint_canvas_array()
        # self.saved_list.value = ''
        self.show_msg('reverted to original')

    def paste(self):
        if self.obj_acts is None:
            self.show_msg('ERR: no activations')
            return
        if self.overwriting or 'paste' not in self.request:
            if self.current_mask_item is None:
                self.show_msg('ERR: no selected item')
                return
            if not self.canvas_array[self.current_mask_item].mask:
                self.show_msg(f'ERR: no mask for selected item at index {self.current_mask_item} (imgnum {self.sel[self.current_mask_item]})')
                return
            imgnum = self.sel[self.current_mask_item]
            mask = self.canvas_array[self.current_mask_item].mask
            self.request['paste'] = (imgnum, mask)
        self.exec_paste()

    def exec_paste(self):
        imgnum, mask = self.request['paste']
        goal_in, goal_out, _, _ = self.gw.paste_from_selection(
            imgnum, mask, self.obj_acts, self.obj_area)
        self.target_out.show([
            'goal (in %d):' % imgnum,
            [self.gw.render_object(self.gw.target_model(goal_in))],
            [self.gw.render_object(goal_out)]])

    def exec_erase(self):
        if 'paste' not in self.request:
            self.show_msg('no paste goal selected...')
            return
        if 'key' not in self.request:
            self.show_msg('no context key selected...')
            return
        self.show_msg('erasing from model...')
        rank = int(self.rank_textbox.value)
        niter = int(self.paste_niter_textbox.value)
        piter = int(self.paste_piter_textbox.value)
        lr = float(self.paste_lr_textbox.value)

        def update_callback(it, loss):
            if it % 50 == 0 or it == niter - 1:
                loss_info = (f'lr {lr:.4f}\titer {it: 6d}/{niter: 6d}'
                             f'\tloss {loss.item():.4f}')
                self.loss_out.print(loss_info, replace=True)
        self.gw.apply_erase(self.request,
                            rank=rank, drank=30, niter=niter, piter=piter, lr=lr,
                            update_callback=update_callback)
        self.repaint_canvas_array()
        self.show_msg(f'erased from model')

    def exec_request(self):
        if 'object' not in self.request:
            self.show_msg('no copy object selected...')
            return
        if 'paste' not in self.request:
            self.show_msg('no paste goal selected...')
            return
        self.show_msg('modifying model...')
        rank = int(self.rank_textbox.value)
        niter = int(self.paste_niter_textbox.value)
        piter = int(self.paste_piter_textbox.value)
        lr = float(self.paste_lr_textbox.value)

        def update_callback(it, loss):
            if it % 50 == 0 or it == niter - 1:
                loss_info = (f'lr {lr:.4f}\titer {it: 6d}/{niter: 6d}'
                             f'\tloss {loss.item():.4f}')
                self.loss_out.print(loss_info, replace=True)
        self.gw.apply_edit(self.request,
                           rank=rank, niter=niter, piter=piter, lr=lr,
                           update_callback=update_callback)
        self.repaint_canvas_array()
        self.show_msg(f'pasted into model')

    def exec_overfit(self):
        if 'object' not in self.request:
            self.show_msg('no copy object selected...')
            return
        if 'paste' not in self.request:
            self.show_msg('no paste goal selected...')
            return
        self.show_msg('overfitting model...')
        niter = int(self.paste_niter_textbox.value)
        lr = float(self.paste_lr_textbox.value)

        def update_callback(it, loss):
            if it % 50 == 0 or it == niter - 1:
                loss_info = (f'lr {lr:.4f}\titer {it: 6d}/{niter: 6d}'
                             f'\tloss {loss.item():.4f}')
                self.loss_out.print(loss_info, replace=True)
        self.gw.apply_overfit(self.request,
                              niter=niter, lr=lr, update_callback=update_callback)
        self.repaint_canvas_array()
        self.show_msg(f'overfitted into model')

    def save(self):
        if not self.saved_list.value:
            return
        self.save_as_name(self.saved_list.value)
        self.saved_list.choices = self.saved_names()
        self.show_msg('saved as ' + self.saved_list.value)

    def tryload(self):
        if not self.saved_list.value:
            return
        if self.saved_list.value in self.saved_list.choices:
            self.show_msg('loading edit...')
            self.load_from_name(self.saved_list.value)
            self.show_msg('loaded from ' + self.saved_list.value + '; exec to execute model change.')

    def saved_names(self):
        os.makedirs(self.savedir, exist_ok=True)
        return sorted([name[:-5] for name in os.listdir(self.savedir)
                       if name.endswith('.json')])

    def save_as_name(self, name):
        data = self.request
        os.makedirs(self.savedir, exist_ok=True)
        with open(os.path.join(self.savedir, '%s.json' % name), 'w') as f:
            json.dump(data, f, indent=1)

    def load_from_name(self, name):
        os.makedirs(self.savedir, exist_ok=True)
        with open(os.path.join(self.savedir, '%s.json' % name), 'r') as f:
            self.request = json.load(f)
            self.repaint_key_tray()
            # if 'query' in self.request:
            #     self.exec_query()
            if 'object' in self.request:
                self.exec_object()
            # self.set_mask(*self.request['paste'])
            if 'paste' in self.request:
                self.exec_paste()

    def show_msg(self, msg):
        # self.msg_out.clear()
        self.msg_out.print(msg, replace=True)

    def widget_html(self):
        return show.html([
            self.object_out,
            self.target_out,
            self.keytray_div,
            self.query_out,
            self.loss_out,
            self.msg_out,
            [self.imgnum_textbox],
            [self.object_btn],
            [self.key_btn],
            [self.query_btn],
            [self.brushsize_textbox],
            [self.highlight_btn],
            [self.snap_btn],
            [self.rank_textbox],
            [self.paste_niter_textbox],
            [self.paste_piter_textbox],
            [self.paste_lr_textbox],
            [self.paste_btn],
            [self.exec_btn],
            [self.erase_btn],
            [self.overfit_btn],
            [self.revert_btn],
            [self.saved_list],
            [self.load_btn],
            [self.save_btn],
            [[c, im] for c, im in
                zip(self.canvas_array, self.snap_image_array)]])


##########################################################################
# Utility functions
##########################################################################

def positive_bounding_box(data):
    pos = (data > 0)
    if pos.sum() == 0:
        return 0, 0, 0, 0
    v, h = pos.sum(0).nonzero(), pos.sum(1).nonzero()
    left, right = v.min().item(), v.max().item()
    top, bottom = h.min().item(), h.max().item()
    return top, left, bottom + 1, right + 1


def centered_location(data):
    t, l, b, r = positive_bounding_box(data)
    return (t + b) // 2, (l + r) // 2


def paste_clip_at_center(source, clip, center, area=None):
    target = source.clone()
    # clip = clip[:,:,:target.shape[2],:target.shape[3]]
    t, l = (max(0, min(e - s, c - s // 2))
            for s, c, e in zip(clip.shape[2:], center, source.shape[2:]))
    b, r = t + clip.shape[2], l + clip.shape[3]
    # TODO: consider copying over a subset of channels.
    target[:, :, t:b, l:r] = clip if area is None else (
        (1 - area)[None, None, :, :].to(target.device) * target[:, :, t:b, l:r] + area[None, None, :, :].to(target.device) * clip)
    return target, (t, l, b, r)


def crop_clip_to_bounds(source, target, bounds):
    t, l, b, r = bounds
    vr, hr = [ts // ss for ts, ss in zip(target.shape[2:], source.shape[2:])]
    st, sl, sb, sr = t // vr, l // hr, -(-b // vr), -(-r // hr)
    tt, tl, tb, tr = st * vr, sl * hr, sb * vr, sr * hr
    cs, ct = source[:, :, st:sb, sl:sr], target[:, :, tt:tb, tl:tr]
    return cs, ct, (st, sl, sb, sr), (tt, tl, tb, tr)


def projected_conv(weight, direction):
    if len(weight.shape) == 5:
        cosine_map = torch.einsum('goiyx, di -> godyx', weight, direction)
        result = torch.einsum('godyx, di -> goiyx', cosine_map, direction)
    else:
        cosine_map = torch.einsum('oiyx, di -> odyx', weight, direction)
        result = torch.einsum('odyx, di -> oiyx', cosine_map, direction)
    return result


def rank_one_conv(weight, direction):
    cosine_map = (weight * direction[None, :, None, None]).sum(1, keepdim=True)
    return cosine_map * direction[None, :, None, None]


def zca_from_cov(cov):
    evals, evecs = torch.symeig(cov.double(), eigenvectors=True)
    zca = torch.mm(torch.mm(evecs, torch.diag
                            (evals.sqrt().clamp(1e-20).reciprocal())),
                   evecs.t()).to(cov.dtype)
    return zca
