
# Usage

    import torch
    from model_3d import Searched_Net_3D
    model = Searched_Net_3D(34).cuda()
    input = torch.rand(32,3,16,112,112).cuda()
    model(input).shape
    
    torch.Size([32, 34])
    


