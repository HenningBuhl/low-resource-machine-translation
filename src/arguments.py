import os

from path_management import CONST_DATA_DIR


# TODO argument names, default values, help strings
# TODO arguments that are not used in argparse but shared in notebooks (e.g. save_top_k or )


# TODO
def sanity_check_args(args):
    pass

def auto_infer_args(args):
    if args.data_dir == None:
        src_tgt_data = os.path.join(CONST_DATA_DIR, f'{args.src_lang}-{args.tgt_lang}')
        tgt_src_data = os.path.join(CONST_DATA_DIR, f'{args.tgt_lang}-{args.src_lang}')
        if os.path.exists(src_tgt_data):
            args.data_dir = src_tgt_data
        elif os.path.exists(tgt_src_data):
            args.data_dir = tgt_src_data
