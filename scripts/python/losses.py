
import torch
import lpips

#gt and pred (NHWC)

class LpipsLoss(torch.nn.Module): #NCHW 

    def __init__(self):
        super(LpipsLoss, self).__init__()
        self.loss_map = torch.zeros([0])
        self.loss_network = lpips.LPIPS().to('cuda')

    def forward(self, pred, gt):
        self.loss_map = self.loss_network(pred.permute(0, 3, 2, 1), gt.permute(0, 3, 2, 1))

        return self.loss_map.mean(),self.loss_map.max()


class DssimL1Loss(torch.nn.Module):

    def __init__(self):
        super(DssimL1Loss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = 2 * torch.abs(pred-gt)
        return self.loss_map.mean()





class SMAPELoss(torch.nn.Module):

    def __init__(self):
        super(SMAPELoss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = torch.abs(pred-gt)/(torch.abs(pred)+torch.abs(gt)+0.01)

        return self.loss_map.mean(),self.loss_map.max()



class DssimL1Loss(torch.nn.Module):

    def __init__(self):
        super(DssimL1Loss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = 2 * torch.abs(pred-gt)
        return self.loss_map.mean(),self.loss_map.max()

class L1Metric(torch.nn.Module):
    def __init__(self):
        super(L1Metric, self).__init__()
        self.loss_map = torch.zeros([0])
        self.max = 0
        self.alpha = 0.5
    def forward(self, predict, gt):
        with torch.no_grad():
            self.loss_map = 2 * torch.abs(predict-gt)
            self.max = self.loss_map.max().item()
        return self.loss_map.mean().item()

    
class RelativeErrorMetric(torch.nn.Module):

    def __init__(self):
        super(RelativeErrorMetric, self).__init__()
        self.loss_map = torch.zeros([0])
        self.max = 0
        self.alpha = 0.5
    def forward(self, predict , gt):
        with torch.no_grad():
            l1 = torch.abs(predict - gt)
            self.loss_map = l1/(torch.min(predict,gt)+0.01)
            self.max = self.loss_map.max().item()
            return self.loss_map.mean().item()
