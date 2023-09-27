import torch
from torch.nn.modules.loss import _Loss

def loss(pred_r, pred_t, pred_c, model_target, model, point_cloud, w, num_point_mesh):
    # get batch size and number of sample points
    bs, num_p, _ = pred_c.size() # (1, 1000, 1)
    
    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1)) # (1, 1000, 4)
    
    base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
    
    # get predicted object points
    base = base.contiguous().transpose(2, 1).contiguous()
    model = model.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    model_target = model_target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    point_cloud = point_cloud.contiguous().view(bs * num_p, 1, 3)
    pred = torch.add(torch.bmm(model, base), point_cloud + pred_t) # (bs*num_p, num_point_mesh, 3)
    
    # get distance and loss
    dis = torch.mean(torch.sum(torch.abs(pred - model_target), dim=2), dim=1)
    
    pred_c = pred_c.contiguous().view(bs * num_p)
    loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)
    
    pred_c = pred_c.view(bs, num_p)
    _, which_max = torch.max(pred_c, 1)
    dis = dis.view(bs, num_p)
    
    return loss, dis[0][which_max[0]]


class ABC_Loss(_Loss):

    def __init__(self, num_points_mesh):
        super(ABC_Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh

    def forward(self, pred_r, pred_t, pred_c, target, model_points, points, w):
        return loss(pred_r, pred_t, pred_c, target, model_points, points, w, self.num_pt_mesh)