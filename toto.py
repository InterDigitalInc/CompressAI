import random
import sys

import torch
from torchvision import transforms

import matplotlib.pyplot as plt

import torch.nn.functional as F
from PIL import Image

from compressai.models.priors import JointAutoregressiveHierarchicalPriors
from compressai import ans

seed = 42
torch.manual_seed(seed)
random.seed(seed)

ckpt_path = '/Users/begaintj/experiments/hosts/lambda-fr/logs/lvc/2020-06-13-jarhp_hbr-0.0250/ckpt_best_loss_model.pth.tar'
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['network']

net = JointAutoregressiveHierarchicalPriors.from_state_dict(state_dict).eval()
net.update()

img = Image.open('/Users/begaintj/data/lena.png').convert('RGB')
# img = img.crop((224, 224, 288, 288))

x = transforms.ToTensor()(img).unsqueeze(0)

with torch.no_grad():
    # Encode
    y = net.g_a(x)
    z = net.h_a(y)

    z_strings = net.entropy_bottleneck.compress(z)
    shape = z.shape[-2:]

    z_hat = net.entropy_bottleneck.decompress(z_strings, shape)
    params = net.h_s(z_hat)

    y_hat = torch.round(y)
    ctx_params = net.context_prediction(y_hat)

    gaussian_params = net.entropy_parameters(
        torch.cat((params, ctx_params), dim=1))
    scales_hat, means_hat = gaussian_params.chunk(2, 1)

    indexes = net.gaussian_conditional.build_indexes(scales_hat)

    def perm(tensor):
        return tensor.permute(0, 2, 3, 1)

    y_strings = net.gaussian_conditional.compress(perm(y_hat),
                                                  perm(indexes),
                                                  means=perm(means_hat))

    # y_hat = net.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)
    x_hat = net.g_s(y_hat).clamp_(0, 1)

    # Decode
    z_dec = net.entropy_bottleneck.decompress(z_strings, shape)
    params_dec = net.h_s(z_dec)

    assert torch.allclose(z_hat, z_dec)
    assert torch.allclose(params, params_dec)

    kernel_size = 5  # context prediction kernel size
    padding = (kernel_size - 1) // 2

    y_dec = torch.zeros((z_hat.size(0), net.M, z_hat.size(2) * 4 + 2 * padding,
                         z_hat.size(3) * 4 + 2 * padding),
                        device=z_hat.device)
    decoder = ans.RangeDecoder()

    decoder.init_decode(y_strings[0])
    cdf = net.gaussian_conditional._quantized_cdf
    cdf = cdf.tolist()
    cdf_lengths = net.gaussian_conditional._cdf_length.reshape(
        -1).int().tolist()
    offsets = net.gaussian_conditional._offset.reshape(-1).int().tolist()

    for h in range(z_dec.size(2) * 4):
        for w in range(z_dec.size(3) * 4):
            print(h, w)

            ctx_params_dec = net.context_prediction(
                torch.round(y_dec[:, :, h:h + kernel_size, w:w + kernel_size]))

            ctx_params_dec = ctx_params_dec[:, :, padding:padding + 1,
                                            padding:padding + 1]

            # print(ctx_params[:, :, h:h + 1, w:w + 1].squeeze()[:10])
            # print(ctx_params_dec.squeeze()[:10])

            assert torch.allclose(ctx_params_dec, ctx_params[:, :, h:h + 1, w:w + 1])

            gaussian_params_dec = net.entropy_parameters(
                torch.cat((params_dec[:, :, h:h + 1, w:w + 1], ctx_params_dec),
                          dim=1))
            scales_hat_dec, means_hat_dec = gaussian_params_dec.chunk(2, 1)

            # assert torch.allclose(scales_hat_dec, scales_hat[:, :, h:h+1, w:w+1])
            # assert torch.allclose(means_hat_dec, means_hat[:, :, h:h+1, w:w+1])

            indexes_dec = net.gaussian_conditional.build_indexes(scales_hat_dec)

            # assert torch.allclose(indexes_dec, indexes[:, :, h:h+1, w:w+1])

            rv = decoder.decode_stream(indexes_dec[0, :].squeeze().int().tolist(),
                                       cdf, cdf_lengths, offsets)

            rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
            rv = net.gaussian_conditional._dequantize(rv, means_hat_dec)

            ref = y_hat[0, :, h:h+1, w:w+1]
            s = torch.abs(rv -ref).sum()
            # assert s == 0, s

            y_dec[:, :, h + padding:h+padding+1, w + padding:w+padding+1] = rv

            # assert torch.allclose(torch.round(y_dec[:, :, h+padding, w+padding]), y_hat[:, :,  h, w])

            # for c in range(320):
            #     a = torch.round(y_dec[0, c, h+padding, w+padding])
            #     b = y_hat[0, c, h, w]
            #     if a != b:
            #         print(c, a-b)

            # if w == 1:
            #     break
        # break

    y_dec = y_dec[:, :, padding:-padding, padding:-padding]
    x_dec = net.g_s(y_dec).clamp_(0, 1)

c = torch.argmax(y_hat[0].std(axis=(1, 2))).item()

# print()
# print(y_hat[0, c])
# print(torch.round(y_dec[0, c]))

fig, axes = plt.subplots(1, 2, figsize=(12, 9))
for ax in axes.ravel():
    ax.axis('off')


axes[0].imshow(y_hat[0, c])
axes[1].imshow(y_dec[0, c])

plt.ion()
plt.show()

rec = transforms.ToPILImage()(x_dec.squeeze())
fig, axes = plt.subplots(1, 3, figsize=(12, 9))
for ax in axes.ravel():
    ax.axis('off')

axes[0].imshow(img)
axes[1].imshow(transforms.ToPILImage()(x_hat.squeeze()))
axes[2].imshow(rec)

plt.ioff()
plt.show()
