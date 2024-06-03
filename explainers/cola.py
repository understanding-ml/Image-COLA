from explainers.pshap import PSHAPExplainer


class ImageCOLA:

    def __init__(self, X, R):
        self.X = X
        self.R = R
        pass

    def compute_matching(self):
        """This corresponds to Lines 1-2 of COLA, with the counterfactual image R given"""
        self.p = None

        # TODO: come up with an algorithm to compute p

    def compute_shapley(self):
        """This corresponds to Lines 3-4 of COLA"""
        self.varphi = None
        pshap_explainer = PSHAPExplainer()

        # TODO: use PSHAPExplainer to compute varphi

    def compute_pixel_values(self):
        """This correspodns to Line 5 of COLA"""
        self.q = None

        # TODO: come up with an algorithm to compute q

    def obtain_counterfactual_image(self, C):
        """This corresponds to Lines 6-16 of COLA"""
        z = None

        # TODO: construct z with self.varphi and self.X, self.R, and self.q
        return z
