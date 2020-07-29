import os
import math
import torch
import numpy as np
import torch.nn.functional as functional
from torch.autograd import Variable
from sklearn.cluster import KMeans
from . import feature_metric as FM
from util import draw_correspondence as draw
from util import util

class sparse_semantic_correspondence():
    def __init__(self, model, gpu_ids, tau, border_size, save_dir, k_per_level, k_final, fast):
        self.Tensor = torch.cuda.FloatTensor if gpu_ids else torch.Tensor
        self.model = model
        self.save_dir = save_dir
        self.border_size = border_size
        self.tau = tau
        self.k_per_level = k_per_level
        self.k_final = k_final
        self.patch_size_list = [[5,5],[5,5],[3,3],[3,3],[3,3]]
        self.search_box_radius_list = [3,3,2,2,2]
        self.draw_radius = [2,2,2,4,8]
        self.pad_mode = 'reflect'
        self.L_final = 2 if fast else 1

    def find_mapping(self, A, B, patch_size, initial_mapping, search_box_radius):
        assert(A.size() == B.size())
        A_to_B_map = self.Tensor(1,2,A.size(2),A.size(3))
        loss_map = self.Tensor(1,1,A.size(2),A.size(3))
        mapping_distance_map = self.Tensor(1,1,A.size(2),A.size(3))
        [dx,dy] = [math.floor(patch_size[0]/2), math.floor(patch_size[1]/2)]
        pad_size = tuple([dy,dy,dx,dx])
        A_padded = functional.pad(A, pad_size, self.pad_mode).data
        B_padded = functional.pad(B, pad_size, self.pad_mode).data

        for i in range(A.size(2)):
            for j in range(A.size(3)):
                init_pix_numpy = initial_mapping[0,:,i,j].cpu().numpy()
                candidate_patch_A = A_padded[:,:,(i):(i+2*dx+1),(j):(j+2*dy+1)]
                index = self.find_closest_patch_index(B_padded, candidate_patch_A, initial_mapping[0,:,i,j], search_box_radius)
                A_to_B_map[:,:,i,j] = self.Tensor([index[0]-dx, index[1]-dy])

        return A_to_B_map

    def find_closest_patch_index(self, B, patch_A, inital_pixel, search_box_radius):
        [dx, dy] = [math.floor(patch_A.size(2)/2), math.floor(patch_A.size(3)/2)]
        [search_dx, search_dy] = [search_box_radius, search_box_radius]
        up_boundary = int(inital_pixel[0]-search_dx) if inital_pixel[0]-search_dx > 0 else 0
        down_boundary = int(inital_pixel[0]+2*dx+search_dx+1) if inital_pixel[0]+2*dx+search_dx+1 < B.size(2) else B.size(2)
        left_boundary = int(inital_pixel[1]-search_dy) if inital_pixel[1]-search_dy > 0 else 0
        right_boundary = int(inital_pixel[1]+2*dy+search_dy+1) if inital_pixel[1]+2*dy+search_dy+1 < B.size(3) else B.size(3)
        search_box_B = B[:,:,up_boundary:down_boundary,left_boundary:right_boundary]
        result_B = functional.conv2d(Variable(search_box_B), Variable(patch_A.contiguous())).data
        distance = result_B
        max_j = distance.max(3)[1]
        max_i = distance.max(3)[0].max(2)[1][0][0]
        max_j = max_j[0,0,max_i]
        closest_patch_distance = distance[0,0,max_i,max_j]
        closest_patch_index = [max_i + dx + up_boundary, max_j + dy + left_boundary]

        return closest_patch_index

    def warp(self, A_size, B, patch_size, mapping):
        assert(B.size() == A_size)
        [dx,dy] = [math.floor(patch_size[0]/2), math.floor(patch_size[1]/2)]
        pad_size = tuple([dy,dy,dx,dx])
        B_padded = functional.pad(B, pad_size, self.pad_mode).data
        warped_A = self.Tensor(B_padded.size()).fill_(0.0)
        counter = self.Tensor(B_padded.size()).fill_(0.0)
        for i in range(A_size[2]):
            for j in range(A_size[3]):
                map_ij = mapping[0,:,i,j]
                warped_A[:,:,i:(i+2*dx+1),j:(j+2*dy+1)] += B_padded[:, :, int(map_ij[0]):(int(map_ij[0])+2*dx+1), int(map_ij[1]):(int(map_ij[1])+2*dy+1)]
                counter[:,:,i:(i+2*dx+1),j:(j+2*dy+1)] += self.Tensor(B_padded.size(0),B_padded.size(1),patch_size[0],patch_size[1]).fill_(1.0)
        return warped_A[:, :, dx:(warped_A.size(2)-dx), dy:(warped_A.size(3)-dy)]/counter[:, :, dx:(warped_A.size(2)-dx), dy:(warped_A.size(3)-dy)]

    def warp_to_mid(self, A_size, B, A, patch_size, mapping):
        assert(B.size() == A_size)
        [dx,dy] = [math.floor(patch_size[0]/2), math.floor(patch_size[1]/2)]
        pad_size = tuple([dy,dy,dx,dx])
        B_padded = functional.pad(B, pad_size, self.pad_mode).data
        A_padded = functional.pad(A, pad_size, self.pad_mode).data
        warped_A = self.Tensor(B_padded.size()).fill_(0.0)
        counter = self.Tensor(B_padded.size()).fill_(0.0)
        for i in range(A_size[2]):
            for j in range(A_size[3]):
                map_ij = mapping[0,:,i,j]
                warped_A[:,:,i:(i+2*dx+1),j:(j+2*dy+1)] += 0.5*(A_padded[:,:,i:(i+2*dx+1),j:(j+2*dy+1)] + B_padded[:, :, int(map_ij[0]):(int(map_ij[0])+2*dx+1), int(map_ij[1]):(int(map_ij[1])+2*dy+1)])
                counter[:,:,i:(i+2*dx+1),j:(j+2*dy+1)] += self.Tensor(B_padded.size(0),B_padded.size(1),patch_size[0],patch_size[1]).fill_(1.0)
        return warped_A[:, :, dx:(warped_A.size(2)-dx), dy:(warped_A.size(3)-dy)]/counter[:, :, dx:(warped_A.size(2)-dx), dy:(warped_A.size(3)-dy)]

    def normalize_0_to_1(self, F):
        assert(F.dim() == 4)
        max_val = F.max()
        min_val = F.min()
        if max_val != min_val:
            F_normalized = (F - min_val)/(max_val-min_val)
        else:
            F_normalized = self.Tensor(F.size()).fill_(0)

        return F_normalized

    def mapping_to_image_size(self, mapping, level, original_image_size):
        if level == 1:
            return mapping
        else:
            identity_map_L =  self.identity_map(mapping.size())
            identity_map_original = self.identity_map(original_image_size)
            factor = int(math.pow(2,level-1))
            return identity_map_original + self.upsample_mapping(mapping - identity_map_L,factor = factor)

    def upsample_mapping(self, mapping, factor = 2):
        upsampler = torch.nn.Upsample(scale_factor=factor, mode='nearest')
        return upsampler(Variable(factor*mapping)).data

    def get_M(self, F, tau=0.05):
        assert(F.dim() == 4)
        F_squared_sum = F.pow(2).sum(1,keepdim=True).expand_as(F)
        F_normalized = self.normalize_0_to_1(F_squared_sum)
        M = self.Tensor(F_normalized.size())
        M.copy_(torch.ge(F_normalized,tau))
        return M

    def identity_map(self, size):
        idnty_map = self.Tensor(size[0],2,size[2],size[3])
        idnty_map[0,0,:,:].copy_(torch.arange(0,size[2]).repeat(size[3],1).transpose(0,1))
        idnty_map[0,1,:,:].copy_(torch.arange(0,size[3]).repeat(size[2],1))
        return idnty_map

    def spatial_distance(self, point_A, point_B):
        return math.pow((point_A - point_B).pow(2).sum(),0.5)

    def find_neural_best_buddies(self, correspondence, F_A, F_Am, F_Bm, F_B, patch_size, initial_map_a_to_b, initial_map_b_to_a, search_box_radius, tau, top, deepest_level=False):
        assert(F_A.size() == F_Bm.size())
        assert(F_Am.size() == F_B.size())

        F_Am_normalized = FM.normalize_per_pix(F_Am)
        F_Bm_normalized = FM.normalize_per_pix(F_Bm)
        a_to_b = self.find_mapping(F_Am_normalized, F_Bm_normalized, patch_size, initial_map_a_to_b, search_box_radius)
        b_to_a = self.find_mapping(F_Bm_normalized, F_Am_normalized, patch_size, initial_map_b_to_a, search_box_radius)

        if deepest_level == True:
            refined_correspondence = self.find_best_buddies(a_to_b, b_to_a)
            refined_correspondence = self.calculate_activations(refined_correspondence, F_A, F_B)
        else:
            refined_correspondence = correspondence
            for i in range(len(correspondence[0])-1,-1,-1):
                [top_left_1, bottom_right_1] = self.extract_receptive_field(correspondence[0][i][0], correspondence[0][i][1], search_box_radius, [a_to_b.size(2), a_to_b.size(3)])
                [top_left_2, bottom_right_2] = self.extract_receptive_field(correspondence[1][i][0], correspondence[1][i][1], search_box_radius, [a_to_b.size(2), a_to_b.size(3)])
                refined_correspondence_i = self.find_best_buddies(a_to_b, b_to_a, top_left_1, bottom_right_1, top_left_2, bottom_right_2)
                refined_correspondence_i = self.calculate_activations(refined_correspondence_i, F_A, F_B)
                refined_correspondence = self.replace_refined_correspondence(refined_correspondence, refined_correspondence_i, i)

        return [refined_correspondence, a_to_b, b_to_a]

    def find_best_buddies(self, a_to_b, b_to_a, top_left_1 = [0,0], bottom_right_1 = [float('inf'), float('inf')], top_left_2 = [0,0], bottom_right_2 = [float('inf'), float('inf')]):
        assert(a_to_b.size() == b_to_a.size())
        correspondence = [[],[]]
        loss = []
        number_of_cycle_consistencies = 0
        for i in range(top_left_1[0], min(bottom_right_1[0],a_to_b.size(2))):
            for j in range(top_left_1[1], min(bottom_right_1[1],a_to_b.size(3))):
                map_ij = a_to_b[0,:,i,j].cpu().numpy() #Should be improved (slow in cuda)
                d = self.spatial_distance(b_to_a[0,:,int(map_ij[0]),int(map_ij[1])],self.Tensor([i,j]))
                if d == 0:
                    if int(map_ij[0]) >= top_left_2[0] and int(map_ij[1]) >= top_left_2[1] and int(map_ij[0]) < bottom_right_2[0] and int(map_ij[1]) < bottom_right_2[1]:
                        correspondence[0].append([i,j])
                        correspondence[1].append([int(map_ij[0]), int(map_ij[1])])
                        number_of_cycle_consistencies += 1
        return correspondence

    def extract_receptive_field(self, x, y, radius, width):
        center = [2*x, 2*y]
        top_left = [max(center[0]-radius, 0), max(center[1]-radius, 0)]
        bottom_right = [min(center[0]+radius+1, width[0]), min(center[1]+radius+1, width[1])]
        return [top_left, bottom_right]

    def replace_refined_correspondence(self, correspondence, refined_correspondence_i, index):
        new_correspondence = correspondence
        activation = correspondence[2][index]
        new_correspondence[0].pop(index)
        new_correspondence[1].pop(index)
        new_correspondence[2].pop(index)

        for j in range(len(refined_correspondence_i[0])):
            new_correspondence[0].append(refined_correspondence_i[0][j])
            new_correspondence[1].append(refined_correspondence_i[1][j])
            new_correspondence[2].append(activation+refined_correspondence_i[2][j])

        return new_correspondence

    def calculate_activations(self, correspondence, F_A, F_B):
        response_A = FM.stretch_tensor_0_to_1(FM.response(F_A))
        response_B = FM.stretch_tensor_0_to_1(FM.response(F_B))
        correspondence_avg_response = self.Tensor(len(correspondence[0])).fill_(0)
        response_correspondence = correspondence
        response_correspondence.append([])
        for i in range(len(correspondence[0])):
            response_A_i = response_A[0,0,correspondence[0][i][0],correspondence[0][i][1]]
            response_B_i = response_B[0,0,correspondence[1][i][0],correspondence[1][i][1]]
            correspondence_avg_response_i = (response_A_i + response_B_i)*0.5
            response_correspondence[2].append(correspondence_avg_response_i)
        return response_correspondence

    def limit_correspondence_number_per_level(self, correspondence, F_A, F_B, tau, top=5):
        correspondence_avg_response = self.Tensor(len(correspondence[0])).fill_(0)
        for i in range(len(correspondence[0])):
            correspondence_avg_response[i] = correspondence[2][i]

        top_response_correspondence = [[],[],[]]
        if len(correspondence[0]) > 0 :
            [sorted_correspondence, ind] = correspondence_avg_response.sort(dim=0, descending=True)
            for i in range(min(top,len(correspondence[0]))):
                #if self.get_M(F_A, tau=tau)[0,0,correspondence[0][ind[i]][0],correspondence[0][ind[i]][1]] == 1 and self.get_M(F_B, tau=tau)[0,0,correspondence[1][ind[i]][0],correspondence[1][ind[i]][1]] == 1:
                top_response_correspondence[0].append(correspondence[0][ind[i]])
                top_response_correspondence[1].append(correspondence[1][ind[i]])
                top_response_correspondence[2].append(sorted_correspondence[i])

        return top_response_correspondence

    def threshold_response_correspondence(self, correspondence, F_A, F_B, th):
        M_A = self.get_M(F_A, tau=th)
        M_Bt = self.get_M(F_B, tau=th)
        high_correspondence = [[],[],[]]
        for i in range(len(correspondence[0])):
            M_A_i = M_A[0,0,correspondence[0][i][0],correspondence[0][i][1]]
            M_Bt_i = M_Bt[0,0,correspondence[1][i][0],correspondence[1][i][1]]
            if M_A_i == 1 and M_Bt_i == 1:
                high_correspondence[0].append(correspondence[0][i])
                high_correspondence[1].append(correspondence[1][i])
                high_correspondence[2].append(correspondence[2][i])

        return high_correspondence

    def make_correspondence_unique(self, correspondence):
        unique_correspondence = correspondence
        for i in range(len(unique_correspondence[0])-1,-1,-1):
            for j in range(i-1,-1,-1):
                if self.is_same_match(unique_correspondence[0][i], unique_correspondence[0][j]):
                    unique_correspondence[0].pop(i)
                    unique_correspondence[1].pop(i)
                    unique_correspondence[2].pop(i)
                    break

        return unique_correspondence

    def remove_border_correspondence(self, correspondence, border_width, image_width):
        filtered_correspondence = correspondence
        for i in range(len(filtered_correspondence[0])-1,-1,-1):
            x_1 = filtered_correspondence[0][i][0]
            y_1 = filtered_correspondence[0][i][1]
            x_2 = filtered_correspondence[1][i][0]
            y_2 = filtered_correspondence[1][i][1]
            if x_1 < border_width or x_1 > image_width - border_width:
                filtered_correspondence[0].pop(i)
                filtered_correspondence[1].pop(i)
                filtered_correspondence[2].pop(i)
            elif x_2 < border_width or x_2 > image_width - border_width:
                filtered_correspondence[0].pop(i)
                filtered_correspondence[1].pop(i)
                filtered_correspondence[2].pop(i)
            elif y_1 < border_width or y_1 > image_width - border_width:
                filtered_correspondence[0].pop(i)
                filtered_correspondence[1].pop(i)
                filtered_correspondence[2].pop(i)
            elif y_2 < border_width or y_2 > image_width - border_width:
                filtered_correspondence[0].pop(i)
                filtered_correspondence[1].pop(i)
                filtered_correspondence[2].pop(i)

        return filtered_correspondence

    def is_same_match(self, corr_1, corr_2):
        if corr_1[0] == corr_2[0] and corr_1[1] == corr_2[1]:
            return True

    def response(self, F):
        response = F.pow(2).sum(1,keepdim=True)
        return response

    def scale_correspondence(self, correspondence, level):
        scaled_correspondence = [[],[],[]]
        scale_factor = int(math.pow(2,level-1))
        for i in range(len(correspondence[0])):
            scaled_correspondence[0].append([scale_factor*correspondence[0][i][0], scale_factor*correspondence[0][i][1]])
            scaled_correspondence[1].append([scale_factor*correspondence[1][i][0], scale_factor*correspondence[1][i][1]])
            scaled_correspondence[2].append(correspondence[2][i])

        return scaled_correspondence

    def save_correspondence_as_txt(self, correspondence, name=''):
        self.save_points_as_txt(correspondence[0], 'correspondence_A' + name)
        self.save_points_as_txt(correspondence[1], 'correspondence_Bt' + name)

    def save_points_as_txt(self, points, name):
        util.mkdirs(self.save_dir)
        file_name = os.path.join(self.save_dir, name + '.txt')
        with open(file_name, 'wt') as opt_file:
            for i in range(len(points)):
                opt_file.write('%i, %i\n' % (points[i][0], points[i][1]))

    def top_k_in_clusters(self, correspondence, k):
        if k > len(correspondence[0]):
            return correspondence

        correspondence_R_4 = []
        for i in range(len(correspondence[0])):
            correspondence_R_4.append([correspondence[0][i][0], correspondence[0][i][1], correspondence[1][i][0], correspondence[1][i][1]])

        top_cluster_correspondence = [[],[],[]]
        print("Calculating K-means...")
        kmeans = KMeans(n_clusters=k, random_state=0).fit(correspondence_R_4)
        for i in range(k):
            max_response = 0
            max_response_index = len(correspondence[0])
            for j in range(len(correspondence[0])):
                if kmeans.labels_[j]==i and correspondence[2][j]>max_response:
                    max_response = correspondence[2][j]
                    max_response_index = j
            top_cluster_correspondence[0].append(correspondence[0][max_response_index])
            top_cluster_correspondence[1].append(correspondence[1][max_response_index])
            top_cluster_correspondence[2].append(correspondence[2][max_response_index])

        return top_cluster_correspondence

    def caculate_mid_correspondence(self, correspondence):
        mid_correspondence = []
        for i in range(len(correspondence[0])):
            x_m = math.floor((correspondence[0][i][0] + correspondence[1][i][0])/2)
            y_m = math.floor((correspondence[0][i][1] + correspondence[1][i][1])/2)
            mid_correspondence.append([x_m, y_m])

        return mid_correspondence

    def transfer_style_local(self, F_A, F_B, patch_size, image_width, mapping_a_to_b, mapping_b_to_a, L):
        F_B_warped = self.warp(F_A.size(), F_B, patch_size, mapping_a_to_b)
        F_A_warped = self.warp(F_B.size(), F_A, patch_size, mapping_b_to_a)
        RL_1B = self.model.deconve(F_B_warped, image_width, L, L-1, print_errors=False).data
        RL_1A = self.model.deconve(F_A_warped, image_width, L, L-1, print_errors=False).data
        self.model.set_input(self.A)
        FL_1A = self.model.forward(level = L-1).data
        self.model.set_input(self.B)
        FL_1B = self.model.forward(level = L-1).data
        FL_1Am = (FL_1A + RL_1B)*0.5
        FL_1Bm = (FL_1B + RL_1A)*0.5
        initial_map_a_to_b = self.upsample_mapping(mapping_a_to_b)
        initial_map_b_to_a = self.upsample_mapping(mapping_b_to_a)
        return [FL_1A, FL_1B, FL_1Am, FL_1Bm, initial_map_a_to_b, initial_map_b_to_a]

    def finalize_correspondence(self, correspondence, image_width, L):
        print("Drawing correspondence...")
        unique_correspondence = self.make_correspondence_unique(correspondence)
        scaled_correspondence = self.scale_correspondence(unique_correspondence, L)
        draw.draw_correspondence(self.A, self.B, scaled_correspondence, self.draw_radius[L-1], self.save_dir, L)
        scaled_correspondence = self.remove_border_correspondence(scaled_correspondence, self.border_size, image_width)
        print("No. of correspondence: ", len(scaled_correspondence[0]))
        return scaled_correspondence

    def run(self, A, B):
        assert(A.size() == B.size())
        image_width = A.size(3)
        print("Saving original images...")
        util.mkdir(self.save_dir)
        util.save_final_image(A, 'original_A', self.save_dir)
        util.save_final_image(B, 'original_B', self.save_dir)
        self.A = self.Tensor(A.size()).copy_(A)
        self.B = self.Tensor(B.size()).copy_(B)
        print("Starting algorithm...")
        L_start = 5

        self.model.set_input(self.A)
        F_A = self.model.forward(level = L_start).data
        self.model.set_input(self.B)
        F_B = self.model.forward(level = L_start).data
        F_Am = F_A.clone()
        F_Bm = F_B.clone()

        initial_map_a_to_b = self.identity_map(F_B.size())
        initial_map_b_to_a = initial_map_a_to_b.clone()

        for L in range(L_start,self.L_final-1,-1):
            patch_size = self.patch_size_list[L-1]
            search_box_radius = self.search_box_radius_list[L-1]
            draw_radius = self.draw_radius[L-1]

            if L == L_start:
                deepest_level = True
                correspondence = []

            else:
                deepest_level = False

            print("Finding best-buddies for the " + str(L) + "-th level")
            [correspondence, mapping_a_to_b, mapping_b_to_a] = self.find_neural_best_buddies(correspondence, F_A, F_Am, F_Bm, F_B, patch_size, initial_map_a_to_b, initial_map_b_to_a, search_box_radius, self.tau, self.k_per_level, deepest_level)
            correspondence = self.threshold_response_correspondence(correspondence, F_A, F_B, self.tau)
            if self.k_per_level < float('inf'):
                correspondence = self.top_k_in_clusters(correspondence, int(self.k_per_level))


            if L > self.L_final:
                print("Drawing correspondence...")
                scaled_correspondence = self.scale_correspondence(correspondence, L)
                draw.draw_correspondence(self.A, self.B, scaled_correspondence, draw_radius, self.save_dir, L)

            [F_A, F_B, F_Am, F_Bm, initial_map_a_to_b, initial_map_b_to_a] = self.transfer_style_local(F_A, F_B, patch_size, image_width, mapping_a_to_b, mapping_b_to_a, L)

        filtered_correspondence = self.finalize_correspondence(correspondence, image_width, self.L_final)
        draw.draw_correspondence(self.A, self.B, filtered_correspondence, self.draw_radius[self.L_final-1], self.save_dir)
        self.save_correspondence_as_txt(filtered_correspondence)
        top_k_correspondence = self.top_k_in_clusters(filtered_correspondence, self.k_final)
        draw.draw_correspondence(self.A, self.B, top_k_correspondence, self.draw_radius[self.L_final-1], self.save_dir, name='_top_'+str(self.k_final))
        self.save_correspondence_as_txt(top_k_correspondence, name='_top_'+str(self.k_final))

        return scaled_correspondence
