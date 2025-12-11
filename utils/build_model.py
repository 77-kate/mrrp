from model.bl_model import *
from model.yx_model import *
from model.model import *
from utils.print_utils import *

def _load_from_checkpoint(opts, model, checkpoint, printer):
    if os.path.isfile(checkpoint):
        state_dict = torch.load(checkpoint, weights_only=True, map_location='cpu')
        model_state_dict = model.state_dict()
        suc = 0
        freezn_keys = []
        for key, value in state_dict.items():
            if opts.finetune:
                if "classifier" in key:
                    continue
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                model_state_dict[key] = value
                freezn_keys.append(key)
                suc += 1
        res = model.load_state_dict(model_state_dict)
        assert(suc == len(model_state_dict))
        print_info_message('Load from {} ({}) {}/{}'.format(checkpoint, res, suc, len(model_state_dict)), printer)
    return model

def build_sub_model(opts, printer=print, model_name='bl'):
    if model_name=='bl':
        model = eval(opts.bl_model)(opts)
        if os.path.isfile(opts.bl_pretrained):
            printer(f'Loaded pretrained weight from {opts.bl_pretrained}')
            state_dict = torch.load(opts.bl_pretrained)
            model_state_dict = model.state_dict()
            suc = 0
            freezn_keys = []
            for key, value in state_dict.items():
                if key in model_state_dict:
                    model_state_dict[key] = value
                    freezn_keys.append(key)
                    suc += 1
            model.load_state_dict(model_state_dict)
            printer(f'Loaded {suc}/{len(list(state_dict.keys()))} keys')  
        model = _load_from_checkpoint(opts, model, opts.bl_checkpoint, printer)
    elif model_name == 'yx':
        model = eval(opts.yx_model)(opts)
        if os.path.isfile(opts.yx_pretrained):
            printer(f'Loaded pretrained weight from {opts.yx_pretrained}')
            state_dict = torch.load(opts.yx_pretrained)
            model_state_dict = model.state_dict()
            suc = 0
            freezn_keys = []
            for key, value in state_dict.items():
                if key in model_state_dict:
                    model_state_dict[key] = value
                    freezn_keys.append(key)
                    suc += 1
            model.load_state_dict(model_state_dict)
            # for key, param in model.named_parameters():
            #     if key in freezn_keys:
            #         param.requires_grad = False
            printer(f'Loaded {suc}/{len(list(state_dict.keys()))} keys')
        model = _load_from_checkpoint(opts, model, opts.yx_checkpoint, printer)
    else:
        print_error_message('SubModel ({}) not yet supported'.format('model_name'), printer)

    return model

def build_model(opts, printer=print):
    model = None
    if opts.modal == 'bl':
        bl_model = build_sub_model(opts, printer=printer, model_name='bl')
        model = eval('BLPIModel')(opts=opts, bl_model=bl_model)
    elif opts.modal == 'yx':
        yx_model = build_sub_model(opts, printer=printer, model_name='yx')
        model = eval('YXPIModel')(opts=opts, yx_model=yx_model)
    elif opts.modal == 'pi':
        model = eval('PIModel')(opts=opts)
    elif opts.modal == 'blyx':
        bl_model = build_sub_model(opts, printer=printer, model_name='bl')
        yx_model = build_sub_model(opts, printer=printer, model_name='yx')
        model = eval(opts.blyx_model)(opts=opts, bl_model=bl_model, yx_model=yx_model)
        model = _load_from_checkpoint(opts, model, opts.blyx_checkpoint, printer)
    else:
        print_error_message('Modal ({}) not yet supported'.format('opts.modal'), printer)

    # sanity check to ensure that everything is fine
    if model is None:
        print_error_message('Model cannot be None. Please check', printer)

    # print_log_message(f"{model} initialized",printer=printer)
    # cnt=1
    # for name, param in model.named_parameters():
    #     print_log_message(f"{name:<40} | {str(param.shape):<20} | {param.requires_grad}",printer=printer)
    #     if cnt < 5:
    #         print_log_message(f"{param}",printer=printer)
    #         cnt+=1
    return model

def get_model_opts(parser):
    group = parser.add_argument_group('Medical Imaging Model Details')

    group.add_argument('--modal', default="blyx", type=str, help='single modal(bingli/yingxiang) or multi-modal(bingli and yingxiang)')
    group.add_argument('--feat-fusion-mode', default="parallel", type=str)
    group.add_argument('--intermodal-fusion', default="mumo", type=str,choices=["cat","sum","mul","mumo"])
    group.add_argument('--use-pi-info', action='store_true', default=False, help='use patient information or not')
    group.add_argument('--use-modal-align', action='store_true', default=False, help='use modal alignment algorithm or not')

    group.add_argument('--yx-model', default="YXModel", type=str, help='Name of YingXiangModel')
    group.add_argument('--yx-pretrained', default='', type=str)
    group.add_argument('--yx-cnn-name', default=None, type=str, help='Name of backbone') # "resnet18"
    group.add_argument('--yx-cnn-pretrained', action='store_true', default=False)
    group.add_argument('--yx-cnn-features', type=int, default=16384,help='Dimension of yx image features extracted by yx-cnn model')
    group.add_argument('--yx-omics-features', type=int, default=93,help='dimension of the yx radiomics features')
    group.add_argument('--yx-clin-features', type=int, default=512,help='dimension of the yx clinical features') 
    group.add_argument('--yx-out-features', type=int, default=512,help='dimension of output features of yx model')
    group.add_argument('--yx-attn-heads', default=1, type=int, help='Number of attention heads') # 2
    group.add_argument('--yx-dropout', default=0.0, type=float, help='Dropout value')
    group.add_argument('--yx-attn-dropout', default=0.0, type=float, help='Dropout value for attention') # 0.2
    group.add_argument('--use-yx-clin', action='store_true', default=False, help='use YX_clinical_report feature or not')
    group.add_argument('--use-yx-rad', action='store_true', default=False, help='use YX_radiomics feature or not')

    group.add_argument('--bl-model', default="BLModel", type=str, help='Name of BingLiModel')
    group.add_argument('--bl-pretrained', default='', type=str) # None
    group.add_argument('--bl-cnn-name', default=None, type=str, help='Name of backbone') # resnet18
    group.add_argument('--bl-bag-features', type=int, default=768, help='dimension of the bl bag-level features') 
    group.add_argument('--bl-omics-features', type=int, default=768,help='dimension of the radiomics features') # 1705
    group.add_argument('--bl-clin-features', type=int, default=768,help='dimension of the WSI clinical features')
    group.add_argument('--bl-out-features', type=int, default=512,help='dimension of output features of bl model')
    group.add_argument('--bl-attn-heads', default=1, type=int, help='Number of attention heads') # 2
    group.add_argument('--bl-dropout', default=0.0, type=float, help='Dropout value')
    group.add_argument('--bl-max-bsz-cnn-gpu0', type=int, default=100, help='Max. batch size on GPU0')
    group.add_argument('--bl-attn-dropout', type=float, default=0.0, help='Proability to drop bag and word attention weights') # 0.2
    group.add_argument('--use-bl-clin', action='store_true', default=False, help='use BL_clinical_report feature or not')
    group.add_argument('--use-bl-rad', action='store_true', default=False, help='use BL_radiomics feature or not')
    
    group.add_argument('--blyx-model', default="BLYXModel", type=str, help='Name of BingLiYingXiangModel')
    group.add_argument('--blyx-out-features', type=int, default=1024, help='Dimension of output features of blyx model')
    group.add_argument('--blyx-attn-heads', default=1, type=int, help='Number of attention heads')
    group.add_argument('--blyx-dropout', default=0.2, type=float, help='Dropout value')

    group.add_argument('--resume', action='store_true', default=False)
    group.add_argument('--keep-best-k-models', type=int, default=-1)

    group.add_argument('--blyx-checkpoint', default='', type=str, help='Checkpoint for resuming')
    group.add_argument('--bl-checkpoint', default='', type=str, help='Checkpoint for resuming')
    group.add_argument('--yx-checkpoint', default='', type=str, help='Checkpoint for resuming')
    return parser