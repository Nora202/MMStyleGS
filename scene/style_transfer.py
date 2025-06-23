import torch
import torch.nn as nn
from scene.gaussian_conv import GaussianConv
from utils.loss_utils import calc_mean_std
import torch.nn.functional as F
from scene.VGG import VGGEncoder, normalize_vgg
import copy
class CNN(nn.Module):
    def __init__(self, matrixSize=32):
        super(CNN,self).__init__()
        # 256x64x64
        self.convs = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128,64,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64,matrixSize,3,1,1))
        # 32x8x8
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
        #self.fc = nn.Linear(32*64,256*256)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        # 32x32
        out = out.view(out.size(0),-1)
        return self.fc(out)

def sample_ode(model,z0, N=10):
    ### NOTE: Use Euler method to sample from the learned flow
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0
    #print(z.shape)
    #z=z.squeeze(2)
    batchsize = z.shape[0]

    traj.append(z)
    for i in range(N):
          t = torch.ones((batchsize,1)) * i / N

          t=t.cuda()

          pred = model(z, t)
          z = z + pred * dt

          traj.append(z)

    return traj

def get_train_tuple(z0=None, z1=None):
    t = torch.rand((z1.shape[0], 1)).cuda()
    z_t = t * z1 + (1. - t) * z0
    target = z1 - z0

    return z_t, t, target

def get_final_tuple(z0=None, z1=None):
    t = torch.ones((z1.shape[0], 1)).cuda()
    z_t = t * z1 + (1. - t) * z0
    target = z1 - z0

    return z_t, t, target


def train_rectified_flow(rectified_flow, z0,z1,inner_iters=500):
    rectified_flow.requires_grad = True
    optimizer = torch.optim.Adam(rectified_flow.parameters(), lr=1e-4)
    for i in range(inner_iters + 1):

        optimizer.zero_grad()
        z_t, t, target = get_train_tuple(z0=z0, z1=z1)

        pred = rectified_flow(z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward(retain_graph=True)

        optimizer.step()## to store the loss curve

    return rectified_flow

class MulLayer(nn.Module):
    def __init__(self, matrixSize=32, adain=True, ode=1,N=100):
        super(MulLayer,self).__init__()
        self.adain = adain
        self.mlp = MLP(512, 512, 512, 6,pretrain=True)
        self.ode=ode
        self.rectified_flow_1 = Flow_MLP()
        self.rectified_flow_2 = Flow_MLP()
        self.rectified_flow_3 = Flow_MLP()
        self.N=N
        self.discriminator = MultiDiscriminator().to(torch.device("cuda:0"))

        if adain:
            return

        self.snet = CNN(matrixSize)
        self.matrixSize = matrixSize

        self.compress = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, matrixSize)
        )
        self.unzip = nn.Sequential(
            nn.Linear(matrixSize, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256)
        )

    def forward(self, cF, sF, clip_feature,from_vgg=False, trans=False,train_step=0,text_inference=False):
        '''
        input:
            point cloud features: [N, C]
            style image features: [1, C, H, W]
            D: matrixSize
        '''
        if text_inference==True:
            cF = cF.T  # [C, N]

            content_mean, content_std = calc_mean_std(cF.unsqueeze(0), 2)  # [1, C, 1]

            style_mean_clip, style_std_clip = self.mlp(clip_feature).contiguous().unsqueeze(2).chunk(2, 1)
            style_mean_clip = style_mean_clip.squeeze(0)
            style_std_clip = style_std_clip.squeeze(0)
            content_mean = content_mean.squeeze(0)
            content_std = content_std.squeeze(0)

            cF = (cF - content_mean) / content_std
            if train_step == 0:
                cF = cF * style_std_clip + style_mean_clip

                return cF.T
            else:
                flow_begin = torch.stack([style_mean_clip, style_std_clip], dim=1)

                flow_begin_embedding = flow_begin.squeeze(2)
                flow_begin_embedding = flow_begin_embedding.T
                if train_step == 1:
                    flow_model = self.rectified_flow_1
                elif train_step == 2:
                    flow_model = self.rectified_flow_2

                elif train_step == 3:
                    flow_model = self.rectified_flow_3

                t = torch.rand((flow_begin_embedding.shape[0], 1)).cuda()
                pred = flow_model(flow_begin_embedding, t)
                traj = sample_ode(flow_model, z0=pred, N=self.N)
                flow_end_pred = traj[-1]

                cF = cF * (flow_end_pred[1:2, :].T) + (flow_end_pred[0:1, :].T)

                return cF.T
        if self.adain:
            cF = cF.T # [C, N]
            style_mean, style_std = calc_mean_std(sF,1) # [1, C, 1] #(1,256,1)
            content_mean, content_std = calc_mean_std(cF.unsqueeze(0),2) # [1, C, 1]
            #print(clip_feature.shape) #(1,512)

            style_mean_clip, style_std_clip = self.mlp(clip_feature).contiguous().unsqueeze(2).chunk(2, 1)
            style_mean_clip= style_mean_clip.squeeze(0)
            style_std_clip = style_std_clip.squeeze(0)
            content_mean = content_mean.squeeze(0)
            content_std = content_std.squeeze(0)


            cF = (cF - content_mean) / content_std
            if train_step==0:
                cF = cF * style_std_clip + style_mean_clip

                loss = self.dis_loss(style_mean_clip, style_mean.squeeze(0)) + self.dis_loss(style_std_clip,
                                                                                             style_std.squeeze(0))

                return cF.T, loss
            else:
                flow_begin = torch.stack([style_mean_clip, style_std_clip], dim=1)
                flow_end  = torch.stack([style_mean.squeeze(0), style_std.squeeze(0)], dim=1)

                flow_begin_embedding = flow_begin.squeeze(2)
                flow_begin_embedding = flow_begin_embedding.T

                flow_end_embedding = flow_end.squeeze(2)
                flow_end_embedding = flow_end_embedding.T
                flow_model=None
                if train_step==1:
                    self.mlp.requires_grad = False
                    flow_model=self.rectified_flow_1
                elif train_step==2:
                    self.rectified_flow_2=copy.deepcopy(self.rectified_flow_1)
                    self.rectified_flow_1.requires_grad = False

                    traj = sample_ode(self.rectified_flow_1, z0=flow_begin_embedding, N=self.N)
                    flow_end_embedding = traj[-1]
                    flow_model = self.rectified_flow_2

                elif train_step==3:
                    self.rectified_flow_3 = copy.deepcopy(self.rectified_flow_2)
                    self.rectified_flow_2.requires_grad = False
                    traj = sample_ode(self.rectified_flow_2, z0=flow_begin_embedding, N=self.N)
                    flow_end_embedding = traj[-1]
                    flow_model = self.rectified_flow_3

                z_t, t, target = get_train_tuple(z0=flow_begin_embedding, z1=flow_end_embedding)

                pred = flow_model(z_t, t)

                loss = self.dis_loss(pred[1:2, :], target[1:2, :]) + self.dis_loss(pred[0:1, :], target[0:1, :])
                traj = sample_ode(flow_model, z0=pred, N=self.N)
                flow_end_pred = traj[-1]

                cF = cF*(flow_end_pred[1:2, :].T) + (flow_end_pred[0:1, :].T)



                return cF.T, loss

        assert cF.size(1) == sF.size(1), 'cF and sF must have the same channel size'
        assert sF.size(0) == 1, 'sF must have batch size 1'
        N, C = cF.size()
        B, C, H, W = sF.size()
        # print(N,C,B,C,H,W) 336342 256 1 256 64 64
        # normalize point cloud features
        cF = cF.T  # [C, N]
        style_mean, style_std = calc_mean_std(sF)  # [1, C, 1]
        content_mean, content_std = calc_mean_std(cF.unsqueeze(0))  # [1, C, 1]

        content_mean = content_mean.squeeze(0)
        content_std = content_std.squeeze(0)

        cF = (cF - content_mean) / content_std  # [C, N]
        # compress point cloud features
        compress_content = self.compress(cF.T).T  # [D, N]

        style_mean_clip, style_std_clip = self.mlp(clip_feature).contiguous().unsqueeze(2).chunk(2, 1)
        # print(style_mean_clip.shape) #(1,256,1)
        style_mean_clip = style_mean_clip.squeeze(0)  # 256,1
        style_std_clip = style_std_clip.squeeze(0)

        # normalize style image features
        sF = sF.view(B, C, -1)
        sF = (sF - style_mean) / style_std  # [1, C, H*W]

        if (trans):

            # get style transformation matrix
            sMatrix = self.snet(sF.reshape(B, C, H, W))  # [B=1, D*D]
            sMatrix = sMatrix.view(self.matrixSize, self.matrixSize)  # [D, D]

            transfeature = torch.mm(sMatrix, compress_content).T  # [N, D]
            out = self.unzip(transfeature).T  # [C, N]

            style_mean = style_mean.squeeze(0)  # [C, 1]
            style_std = style_std.squeeze(0)  # [C, 1]

            style_mean_clip = style_mean_clip.squeeze(0)  # [C, 1]
            style_std_clip = style_std_clip.squeeze(0)  # [C, 1]
        else:
            out = self.unzip(compress_content.T)  # [N, C]
            out = out * content_std + content_mean
            return out

    def dis_loss(self, vgg, clip):

        loss = F.mse_loss(vgg, clip)

        return loss

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,pretrain=True) -> None:
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        if pretrain==True:

            state_dict = torch.load('MLP2_iter_160000.pth')
            state_dict.pop('layers.5.weight')
            state_dict.pop('layers.5.bias')
            self.load_state_dict(state_dict, strict=False)
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x.float())) if i < self.num_layers - 1 else layer(x.float())
        return x


class Flow_MLP(nn.Module):
    def __init__(self, input_dim=256, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim+1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        #self.act = torch.tanh()

    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1)

        x = self.fc1(inputs)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = nn.Sequential(*list(VGG.children())[:31])
        self.cnn = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3), nn.ReLU(),
                                   nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3), nn.ReLU(),
                                   nn.AdaptiveAvgPool2d(1)])
        self.fc = nn.Sequential(*[nn.Linear(1024, 1024), nn.ReLU(),
                                  nn.Linear(1024, 1), nn.Sigmoid()])

    def get_patch(self, img):
        ret = []
        for b in range(img.shape[0]):
            for _ in range(4):
                x, y = np.random.randint(0, img.shape[2] - 32), np.random.randint(0, img.shape[3] - 32)
                ret.append(img[b, :, x:x + 32, y:y + 32].unsqueeze(0))
        ret = torch.cat(ret, dim=0)
        return ret

    def forward(self, patch, ins):
        f = self.enc(patch)
        f = self.cnn(f).squeeze()
        out = self.fc(torch.cat([f, ins], dim=1))

        return out


class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs
