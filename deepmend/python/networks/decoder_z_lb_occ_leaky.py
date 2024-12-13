import torch
import torch.nn as nn
import torch.nn.functional as F


def update_args(specs, **kwargs):
    specs_args = specs["SubnetSpecs"]
    specs_args.update(kwargs)
    return specs_args


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        tool_latent_size,
        dims,
        num_dims=3,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        latent_in_inflate=False,
        weight_norm=False,
        xyz_in_all=False,
        use_tanh=False,
        latent_dropout=False,
        do_code_regularization=False,
        use_occ=True,
        subnet_dims=None,
        subnet_dropout=None,
        subnet_norm=None,
        subnet_xyz=False,
        subnet_latent_in=(),
        subnet_latent_in_inflate=False,
    ):
        super(Decoder, self).__init__()

        assert use_occ

        dims = [latent_size + num_dims] + dims + [1]
        for k in latent_in:    ######adapt vector length for skip connection
    	    dims[k]= dims[k]+latent_size #+  num_dims
        self.latent_size = latent_size
        self.tool_latent_size = tool_latent_size
        self.num_dims = num_dims
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.weight_norm = weight_norm
        self.pts_in_all = xyz_in_all
        self.use_tanh = use_tanh
        self.do_code_regularization = do_code_regularization
        self.latent_dropout = latent_dropout
        self.subnet_xyz = subnet_xyz

        self.num_layers = len(dims)
        subnet_pts_in = 0
        if subnet_xyz:
            subnet_pts_in = self.num_dims

        # Build hnet
        hnet_dims = [tool_latent_size + subnet_pts_in] + subnet_dims + [1]
        self.hnet_dropout = subnet_dropout
        self.hnet_norm_layers = subnet_norm
        self.hnet_num_layers = len(hnet_dims)
        for layer in range(0, self.hnet_num_layers - 1):
            if subnet_latent_in_inflate:
                out_dim = hnet_dims[layer + 1]
                if layer in subnet_latent_in:
                    in_dim = hnet_dims[layer] + hnet_dims[0]
                else:
                    in_dim = hnet_dims[layer]
            else:
                in_dim = hnet_dims[layer]
                if layer + 1 in subnet_latent_in:
                    out_dim = hnet_dims[layer + 1] - dims[0]
                else:
                    out_dim = hnet_dims[layer + 1]
            if (self.pts_in_all) and (layer != self.num_layers - 2):
                out_dim -= self.num_dims

            if weight_norm and (layer in self.hnet_norm_layers):
                setattr(
                    self,
                    "hnet_lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(hnet_dims[layer], out_dim)),
                )
            else:
                setattr(
                    self, "hnet_lin" + str(layer), nn.Linear(hnet_dims[layer], out_dim)
                )

            if (
                (not weight_norm)
                and (self.hnet_norm_layers is not None)
                and (layer in self.hnet_norm_layers)
            ):
                setattr(self, "hnet_bn" + str(layer), nn.LayerNorm(out_dim))

        # Build fnet
        for layer in range(0, self.num_layers - 1):
            if latent_in_inflate:
                out_dim = dims[layer + 1]
                if layer in latent_in:
                    in_dim = dims[layer] + dims[0]
                else:
                    in_dim = dims[layer]
            else:
                in_dim = dims[layer]
                if layer + 1 in latent_in:
                    out_dim = dims[layer + 1] - dims[0]
                else:
                    out_dim = dims[layer + 1]
            if (self.pts_in_all) and (layer != self.num_layers - 2):
                out_dim -= self.num_dims

            if weight_norm and (layer in norm_layers):
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(in_dim, out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(in_dim, out_dim))

            if (
                (not weight_norm)
                and (norm_layers is not None)
                and (layer in norm_layers)
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()

        if not use_occ:
            self.th = nn.Tanh()

        setattr(
            self,
            "secondary_net_parameters",
            [p for n, p in self.named_parameters() if ("hnet" in n)],
        )
        setattr(
            self,
            "primary_net_parameters",
            [p for n, p in self.named_parameters() if ("hnet" not in n)],
        )

        self._ones = None

        #for name, param in self.named_parameters():
        #    print(f"{name}: {param.numel()}")


        
    def disable_f_grad(self):
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            lin.requires_grad = False

            if layer < self.num_layers - 2:
                if (
                    (self.norm_layers is not None)
                    and (layer in self.norm_layers)
                    and (not self.weight_norm)
                ):
                    bn = getattr(self, "bn" + str(layer))
                    bn.requires_grad = False

    def reset_ones(self, num_pts):
        self._ones = torch.ones((num_pts, 1)).cuda()
        self._ones.requires_grad = False

    def forward(self, net_input, use_net=None):
        # Input is N x (|z|+|bz|+num_dims)
        latent_vecs = net_input[:, : self.latent_size]
        break_latent_vecs = net_input[:, self.latent_size : -self.num_dims]
        pts = net_input[:, -self.num_dims :]

        # Apply dropout to the latent vector directly
        if net_input.shape[1] > 2 and self.latent_dropout:
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            break_latent_vecs = F.dropout(
                break_latent_vecs, p=0.2, training=self.training
            )

        if (self._ones is None) or (self._ones.shape[0] != net_input.shape[0]):
            self.reset_ones(net_input.shape[0])

        h_input = torch.cat([break_latent_vecs, pts], dim=1)
        f_input = torch.cat([latent_vecs, pts], dim=1)

        def ff_fnet(x):
            # Store the latent vector for skip connections
            x_input = x

            # Feed forward Bnet
            for layer in range(0, self.num_layers - 1):
                # Latent vector appended to input
                if layer in self.latent_in:
                    x = torch.cat([x, x_input], 1)

                # pts appended to input
                elif layer != 0 and self.pts_in_all:
                    x = torch.cat([x, pts], 1)

                # Feed forward
                lin = getattr(self, "lin" + str(layer))
                x = lin(x)

                # Apply tanh layer (second to last)
                if (layer == self.num_layers - 2) and self.use_tanh:
                    x = self.tanh(x)

                if layer < self.num_layers - 2:
                    # Apply weight normalization
                    if (
                        (self.norm_layers is not None)
                        and (layer in self.norm_layers)
                        and (not self.weight_norm)
                    ):
                        bn = getattr(self, "bn" + str(layer))
                        x = bn(x)

                    # Apply relu
                    x = self.relu(x)

                    # Apply dropout
                    if (self.dropout is not None) and (layer in self.dropout):
                        x = F.dropout(x, p=self.dropout_prob, training=self.training)

            # Apply final layer
            if hasattr(self, "th"):
                x = self.th(x)

            return x

        def ff_hnet(x):
            # Feed forward hnet
            for layer in range(0, self.hnet_num_layers - 1):
                # Feed forward
                lin = getattr(self, "hnet_lin" + str(layer))
                x = lin(x)

                if layer < self.hnet_num_layers - 1:
                    # Apply weight normalization
                    if (
                        (self.hnet_norm_layers is not None)
                        and (layer in self.hnet_norm_layers)
                        and (not self.weight_norm)
                    ):
                        bn = getattr(self, "hnet_bn" + str(layer))
                        x = bn(x)

                    # Apply relu
                    x = self.relu(x)

                    # Apply dropout
                    if (self.hnet_dropout is not None) and (layer in self.hnet_dropout):
                        x = F.dropout(x, p=self.dropout_prob, training=self.training)

            return x

        if use_net is None:
            # Feed forward hnet
            t_x = ff_hnet(h_input)

            # Feed forward fnet
            c_x = ff_fnet(f_input)

            # b_x and r_x are not logits anymore
            t_x_sig, c_x_sig = torch.sigmoid(t_x), torch.sigmoid(c_x)
            b_x = c_x_sig * (self._ones - t_x_sig)
            r_x = c_x_sig * t_x_sig

            return (c_x, b_x, r_x, t_x)
        elif use_net == 0:
            return ff_fnet(f_input)
        elif use_net == 1:
            t_x = ff_hnet(h_input)
            c_x = ff_fnet(f_input)
            t_x_sig, c_x_sig = torch.sigmoid(t_x), torch.sigmoid(c_x)
            return c_x_sig * (self._ones - t_x_sig)
        elif use_net == 2:
            t_x = ff_hnet(h_input)
            c_x = ff_fnet(f_input)
            t_x_sig, c_x_sig = torch.sigmoid(t_x), torch.sigmoid(c_x)
            return c_x_sig * t_x_sig
        elif use_net == 3:
            return ff_hnet(h_input)
        else:
            raise RuntimeError(
                "Requested output from non-existent network: {}".format(use_net)
            )
