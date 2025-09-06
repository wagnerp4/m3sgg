from argparse import ArgumentParser

BATCHNORM_MOMENTUM = 0.01


class Config(object):
    """Configuration class for scene graph generation model training and evaluation.

    This class manages all configuration parameters for different model architectures
    including STTRAN, STKET, Tempura, EASG, and SceneLLM. It handles command-line
    argument parsing and sets appropriate defaults for training, evaluation, and
    inference modes.

    :param object: Base object class
    :type object: class
    """

    def __init__(self):
        """Initialize the configuration object with default parameters and command-line argument parsing.

        Sets up all default model parameters, dataset paths, training hyperparameters,
        and model-specific configurations for different architectures including STTRAN,
        STKET, Tempura, EASG, and SceneLLM.

        :return: None
        :rtype: None
        """
        self.mode = None
        self.save_path = None
        self.model_path = None
        self.dataset = None
        self.data_path = None
        self.datasize = None
        self.fraction = 1
        self.ckpt = None
        self.optimizer = None
        self.bce_loss = False
        self.lr = 1e-5
        self.enc_layer = 1
        self.dec_layer = 3
        self.nepoch = 10
        self.device = "cuda:0"
        self.seed = 42
        self.num_workers = 0

        # My parameters
        self.use_matcher = False  # TODO: remove
        self.model_type = "sttran"  # Default model type

        # STKET specific parameters
        self.enc_layer_num = 1
        self.dec_layer_num = 3
        self.N_layer = 1
        self.pred_contact_threshold = 0.5
        self.window_size = 3
        self.use_spatial_prior = False
        self.use_temporal_prior = False
        self.spatial_prior_loss = False
        self.temporal_prior_loss = False
        self.eval = False

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)

        # Tempura specific parameters
        if self.mem_feat_lambda is not None:
            self.mem_feat_lambda = float(self.mem_feat_lambda)
        if self.rel_mem_compute == "None":
            self.rel_mem_compute = None
        if self.obj_loss_weighting == "None":
            self.obj_loss_weighting = None
        if self.rel_loss_weighting == "None":
            self.rel_loss_weighting = None
        self.obj_head = "gmm"
        self.rel_head = "gmm"
        self.K = 4
        self.rel_mem_compute = None
        self.obj_mem_compute = False
        self.take_obj_mem_feat = False
        self.obj_mem_weight_type = "simple"
        self.rel_mem_weight_type = "simple"
        self.mem_feat_selection = "manual"
        self.mem_fusion = "early"
        self.mem_feat_lambda = None
        self.pseudo_thresh = 7
        self.obj_unc = False
        self.rel_unc = False
        self.obj_loss_weighting = None
        self.rel_loss_weighting = None
        self.mlm = False
        self.eos_coef = 1
        self.obj_con_loss = None
        self.lambda_con = 1
        self.tracking = True

        # SceneLLM specific defaults
        self.embed_dim = 1024
        self.codebook_size = 8192
        self.commitment_cost = 0.25
        self.llm_name = "google/gemma-2-2b"
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.ot_step = 512
        self.vqvae_epochs = 5
        self.stage1_iterations = 30000
        self.stage2_iterations = 50000
        self.alpha_obj = 1.0
        self.alpha_rel = 1.0

        # OED specific defaults
        self.num_queries = 100
        self.dec_layers_hopd = 6
        self.dec_layers_interaction = 6
        self.num_attn_classes = 3
        self.num_spatial_classes = 6
        self.num_contacting_classes = 17
        self.alpha = 0.5
        self.oed_use_matching = True
        self.bbox_loss_coef = 2.5
        self.giou_loss_coef = 1.0
        self.obj_loss_coef = 1.0
        self.rel_loss_coef = 2.0
        self.oed_eos_coef = 0.1
        self.interval1 = 4
        self.interval2 = 4
        self.num_ref_frames = 2
        self.oed_variant = "multi"
        self.fuse_semantic_pos = False
        self.query_temporal_interaction = False

        # VLM specific defaults
        self.vlm_model_name = "Salesforce/blip-vqa-base" # Salesforce/blip-vqa-base, apple/FastVLM-0.5B
        self.vlm_use_chain_of_thought = True
        self.vlm_use_tree_of_thought = False
        self.vlm_confidence_threshold = 0.5
        self.vlm_temperature = 0.7
        self.vlm_top_p = 0.9
        self.vlm_max_new_tokens = 512
        self.oed_weight_dict = {
            "loss_obj_ce": self.obj_loss_coef,
            "loss_bbox": self.bbox_loss_coef,
            "loss_giou": self.giou_loss_coef,
            "loss_attn_ce": self.rel_loss_coef,
            "loss_spatial_ce": self.rel_loss_coef,
            "loss_contacting_ce": self.rel_loss_coef,
        }
        self.oed_losses = [
            "obj_labels",
            "boxes",
            "attn_labels",
            "spatial_labels",
            "contacting_labels",
        ]

        # Matcher logic TODO: remove this
        if self.model_type == "dsg-detr":  # model == introduction of matcher.
            self.use_matcher = True
        if self.model_type == "sttran":
            self.use_matcher = False
        elif self.model_type == "stket":
            self.use_matcher = False
            self.enc_layer_num = self.enc_layer
            self.dec_layer_num = self.dec_layer
        elif self.model_type == "EASG":
            self.use_matcher = False
        elif self.model_type == "scenellm":
            self.use_matcher = False
        elif self.model_type == "oed":
            self.use_matcher = self.oed_use_matching

    def setup_parser(self):
        """Set up command-line argument parser for training configuration.

        Creates and configures an ArgumentParser with all available command-line
        options for model training, including dataset selection, model type,
        hyperparameters, and architecture-specific settings.

        :return: Configured ArgumentParser instance
        :rtype: ArgumentParser
        """
        parser = ArgumentParser(description="training code")
        parser.add_argument(
            "-mode",
            dest="mode",
            help="predcls/sgcls/sgdet",
            default="predcls",
            type=str,
        )
        parser.add_argument("-save_path", default="output", type=str)
        parser.add_argument("-model_path", default="weights/predcls.tar", type=str)
        parser.add_argument(
            "-dataset",
            choices=["action_genome", "EASG"],
            default="action_genome",
            type=str,
        )
        parser.add_argument("-data_path", default="data/action_genome", type=str)
        parser.add_argument(  # TODO: use this for subset sampling
            "-datasize",
            dest="datasize",
            help="mini dataset or whole",
            default="large",
            type=str,
        )
        parser.add_argument(
            "-fraction",
            dest="fraction",
            help="Fraction of dataset to use (1=all, 2=half, 4=quarter, etc.)",
            default=1,
            type=int,
        )
        parser.add_argument(
            "-ckpt", dest="ckpt", help="checkpoint", default=None, type=str
        )
        parser.add_argument(
            "-optimizer", help="adamw/adam/sgd", default="adamw", type=str
        )
        parser.add_argument(
            "-lr", dest="lr", help="learning rate", default=1e-5, type=float
        )
        parser.add_argument("-nepoch", help="epoch number", default=10, type=float)
        parser.add_argument(
            "-enc_layer",
            dest="enc_layer",
            help="spatial encoder layer",
            default=1,
            type=int,
        )
        parser.add_argument(
            "-dec_layer",
            dest="dec_layer",
            help="temporal decoder layer",
            default=3,
            type=int,
        )
        parser.add_argument(
            "-device",
            dest="device",
            help="torch device string (e.g., cuda:0, cpu)",
            default="cuda:0",
            type=str,
        )
        parser.add_argument(
            "-seed",
            dest="seed",
            help="global random seed",
            default=42,
            type=int,
        )
        parser.add_argument(
            "-num_workers",
            dest="num_workers",
            help="number of DataLoader workers",
            default=0,
            type=int,
        )
        parser.add_argument(
            "-bce_loss",
            dest="bce_loss",
            help="use BCE loss instead of multi-label margin loss",
            action="store_true",
        )
        parser.add_argument(
            "-model",
            dest="model_type",
            help="Model type: sttran (default), dsg-detr (uses Hungarian matcher), stket, tempura, scenellm, oed, or vlm",
            choices=[
                "sttran",
                "dsg-detr",
                "stket",
                "easg",
                "tempura",
                "scenellm",
                "oed",
                "vlm",
            ],
            default="sttran",
            type=str,
        )

        # STKET specific arguments
        parser.add_argument("-N_layer", default=1, type=int)
        parser.add_argument("-pred_contact_threshold", default=0.5, type=float)
        parser.add_argument("-window_size", default=3, type=int)
        parser.add_argument("-use_spatial_prior", action="store_true")
        parser.add_argument("-use_temporal_prior", action="store_true")
        parser.add_argument("-spatial_prior_loss", action="store_true")
        parser.add_argument("-temporal_prior_loss", action="store_true")
        parser.add_argument("-eval", action="store_true")

        # Tempura specific arguments
        parser.add_argument(
            "-obj_head", default="gmm", type=str, help="classification head type"
        )
        parser.add_argument(
            "-rel_head", default="gmm", type=str, help="classification head type"
        )
        parser.add_argument("-K", default=4, type=int, help="number of mixture models")
        # Memory arguments
        parser.add_argument(
            "-rel_mem_compute",
            default=None,
            type=str,
            help="compute relation memory hallucination [seperate/joint/None]",
        )
        parser.add_argument("-obj_mem_compute", action="store_true")
        parser.add_argument("-take_obj_mem_feat", action="store_true")
        parser.add_argument(
            "-obj_mem_weight_type",
            default="simple",
            type=str,
            help="type of memory [both/al/ep/simple]",
        )
        parser.add_argument(
            "-rel_mem_weight_type",
            default="simple",
            type=str,
            help="type of memory [both/al/ep/simple]",
        )
        parser.add_argument(
            "-mem_feat_selection", default="manual", type=str, help="manual/automated"
        )
        parser.add_argument("-mem_fusion", default="early", type=str, help="early/late")
        parser.add_argument(
            "-mem_feat_lambda", default=None, type=str, help="selection lambda"
        )
        parser.add_argument(
            "-pseudo_thresh", default=7, type=int, help="pseudo label threshold"
        )
        # uncertainty arguments
        parser.add_argument("-obj_unc", action="store_true")
        parser.add_argument("-rel_unc", action="store_true")
        # loss arguments
        parser.add_argument(
            "-obj_loss_weighting", default=None, type=str, help="ep/al/None"
        )
        parser.add_argument(
            "-rel_loss_weighting", default=None, type=str, help="ep/al/None"
        )
        parser.add_argument("-mlm", action="store_true")
        parser.add_argument(
            "-eos_coef",
            default=1,
            type=float,
            help="background class scaling in ce or nll loss",
        )
        parser.add_argument(
            "-obj_con_loss",
            default=None,
            type=str,
            help="intra video visual consistency loss for objects (euc_con/info_nce)",
        )
        parser.add_argument(
            "-lambda_con", default=1, type=float, help="visual consistency loss coef"
        )

        # SceneLLM specific arguments
        parser.add_argument(
            "-embed_dim", default=1024, type=int, help="embedding dimension for VQ-VAE"
        )
        parser.add_argument(
            "-codebook_size", default=8192, type=int, help="size of VQ-VAE codebook"
        )
        parser.add_argument(
            "-commitment_cost",
            default=0.25,
            type=float,
            help="commitment cost for VQ-VAE",
        )
        parser.add_argument(
            "-llm_name", default="google/gemma-2-2b", type=str, help="LLM model name"
        )
        parser.add_argument("-lora_r", default=16, type=int, help="LoRA rank")
        parser.add_argument("-lora_alpha", default=32, type=int, help="LoRA alpha")
        parser.add_argument(
            "-lora_dropout", default=0.05, type=float, help="LoRA dropout"
        )
        parser.add_argument(
            "-ot_step",
            default=512,
            type=int,
            help="step size for optimal transport codebook update",
        )
        parser.add_argument(
            "-vqvae_epochs", default=5, type=int, help="epochs for VQ-VAE pretraining"
        )
        parser.add_argument(
            "-stage1_iterations",
            default=30000,
            type=int,
            help="iterations for stage 1 training",
        )
        parser.add_argument(
            "-stage2_iterations",
            default=50000,
            type=int,
            help="iterations for stage 2 training",
        )
        parser.add_argument(
            "-alpha_obj",
            default=1.0,
            type=float,
            help="weight for object loss in SceneLLM",
        )
        parser.add_argument(
            "-alpha_rel",
            default=1.0,
            type=float,
            help="weight for relation loss in SceneLLM",
        )
        parser.add_argument(
            "-scenellm_training_stage",
            default="vqvae",
            type=str,
            choices=["vqvae", "stage1", "stage2"],
            help="SceneLLM training stage",
        )
        parser.add_argument(
            "-disable_checkpoint_saving",
            action="store_true",
            help="disable all checkpoint saving to save local storage space",
        )

        # OED specific arguments
        parser.add_argument(
            "-num_queries", default=100, type=int, help="Number of query slots for OED"
        )
        parser.add_argument(
            "-dec_layers_hopd",
            default=6,
            type=int,
            help="Number of hopd decoding layers in OED transformer",
        )
        parser.add_argument(
            "-dec_layers_interaction",
            default=6,
            type=int,
            help="Number of interaction decoding layers in OED transformer",
        )
        parser.add_argument(
            "-num_attn_classes", default=3, type=int, help="Number of attention classes"
        )
        parser.add_argument(
            "-num_spatial_classes",
            default=6,
            type=int,
            help="Number of spatial classes",
        )
        parser.add_argument(
            "-num_contacting_classes",
            default=17,
            type=int,
            help="Number of contacting classes",
        )
        parser.add_argument(
            "-alpha", default=0.5, type=float, help="Focal loss alpha for OED"
        )
        parser.add_argument(
            "-oed_use_matching",
            action="store_true",
            help="Use obj/sub matching 2class loss in OED decoder",
        )
        parser.add_argument(
            "-bbox_loss_coef", default=2.5, type=float, help="L1 box coefficient"
        )
        parser.add_argument(
            "-giou_loss_coef", default=1, type=float, help="GIoU box coefficient"
        )
        parser.add_argument(
            "-obj_loss_coef",
            default=1,
            type=float,
            help="Object classification coefficient",
        )
        parser.add_argument(
            "-rel_loss_coef",
            default=2,
            type=float,
            help="Relation classification coefficient",
        )
        parser.add_argument(
            "-oed_eos_coef",
            default=0.1,
            type=float,
            help="Relative classification weight of no-object class for OED",
        )
        parser.add_argument(
            "-interval1",
            default=4,
            type=int,
            help="Interval for training frame selection",
        )
        parser.add_argument(
            "-interval2", default=4, type=int, help="Interval for test frame selection"
        )
        parser.add_argument(
            "-num_ref_frames", default=2, type=int, help="Number of reference frames"
        )
        parser.add_argument(
            "-oed_variant",
            default="multi",
            type=str,
            choices=["single", "multi"],
            help="OED variant: single frame or multi frame",
        )
        parser.add_argument(
            "-fuse_semantic_pos",
            action="store_true",
            help="Fuse semantic and positional embeddings",
        )
        parser.add_argument(
            "-query_temporal_interaction",
            action="store_true",
            help="Enable query temporal interaction",
        )

        # VLM specific arguments
        parser.add_argument(
            "-vlm_model_name",
            default="apple/FastVLM-0.5B",
            type=str,
            help="HuggingFace model name for VLM",
        )
        parser.add_argument(
            "-vlm_use_chain_of_thought",
            action="store_true",
            help="Use chain-of-thought reasoning for VLM",
        )
        parser.add_argument(
            "-vlm_use_tree_of_thought",
            action="store_true",
            help="Use tree-of-thought reasoning for VLM",
        )
        parser.add_argument(
            "-vlm_confidence_threshold",
            default=0.5,
            type=float,
            help="Confidence threshold for VLM relationship detection",
        )
        parser.add_argument(
            "-vlm_temperature",
            default=0.7,
            type=float,
            help="Temperature for VLM text generation",
        )
        parser.add_argument(
            "-vlm_top_p",
            default=0.9,
            type=float,
            help="Top-p sampling for VLM text generation",
        )
        parser.add_argument(
            "-vlm_max_new_tokens",
            default=512,
            type=int,
            help="Maximum number of new tokens for VLM generation",
        )

        return parser
    
    def parse_args(self):
        """Parse command-line arguments and update config values.
        
        :return: None
        :rtype: None
        """
        parser = self.setup_parser()
        args = parser.parse_args()
        
        # Update config with parsed arguments
        for arg_name, arg_value in vars(args).items():
            if hasattr(self, arg_name):
                setattr(self, arg_name, arg_value)
