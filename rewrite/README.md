## Rewriting a Deep Generative Model

 - ganrewrite contains the method.  It contains two Rewriter classes
   (one for Progressive GAN and another for StyleGANv2), which can
   split up a network and directly operate on the weights to change
   rules according to a user specification.  It can be used without
   UI for benchmarking, or within the UI.

 - rewriteapp contains the interactive application as a labwidget
   prototype, suitable for running inside a notebook.
