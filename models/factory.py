# Description: Factory for models
def get_model_loader(args):
    if args.method == 'codyra':
        from models.codyra import get_codyra
        return get_codyra
    else:
        raise ValueError(f"Unknown method: {args.method}")

def get_trainer(args):
    if args.method == 'codyra':
        from models.codyra import finetune
        return finetune
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
