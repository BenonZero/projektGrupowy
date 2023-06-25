import torch


class DeepTemplateMatchingModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 1st module'
        self.evaluation_layers = torch.nn.ModuleList()
        self.template_layers = torch.nn.ModuleList()

        self.evaluation_layers.append(torch.nn.Conv2d(1, 16, 5))
        self.template_layers.append(torch.nn.Conv2d(1, 16, 5))

        self.evaluation_layers.append(torch.nn.Conv2d(16, 32, 5))
        self.template_layers.append(torch.nn.Conv2d(16, 32, 5))

        self.evaluation_layers.append(torch.nn.Conv2d(32, 64, 5))
        self.template_layers.append(torch.nn.Conv2d(32, 64, 5))

        self.evaluation_layers.append(torch.nn.MaxPool2d((2, 5), (2, 1)))
        self.template_layers.append(torch.nn.MaxPool2d((2, 5), (2, 1)))

        self.evaluation_linear = torch.nn.Linear(3712, 64)
        self.template_linear = torch.nn.Linear(3712, 64)

        self.evaluation_gru = torch.nn.GRU(64, 64)
        self.template_gru = torch.nn.GRU(64, 64)
        # 2nd module
        self.attention = torch.nn.Linear(64, 1)

        # 3rd module
        self.hidden = torch.nn.Linear(64, 128)
        self.activation = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(128, 2)

    def forward(self, evaluation, template):
        evaluation = torch.unsqueeze(evaluation, 1)
        template = torch.unsqueeze(template, 1)

        for layer in self.evaluation_layers:
            evaluation = layer(evaluation)
        if len(evaluation.size()) > 3:
            evaluation = torch.reshape(evaluation, (evaluation.size(0), evaluation.size(-1), 3712))
        else:
            evaluation = torch.reshape(evaluation, (evaluation.size(-1), 3712))
        evaluation = self.evaluation_linear(evaluation)
        evaluation, _ = self.evaluation_gru(evaluation)

        for layer in self.template_layers:
            template = layer(template)
        if len(template.size()) > 3:
            template = torch.reshape(template, (template.size(0), template.size(-1), 3712))
        else:
            template = torch.reshape(template, (template.size(-1), 3712))
        template = self.template_linear(template)
        template, _ = self.template_gru(template)

        # end of first module

        # equation (3),(4)
        '''
        # tt = []
        # if len(evaluation.size()) > 2:
        #     for i in range(evaluation.size()[0]):
        #         tt.append(torch.tensordot(evaluation[i], template[i], dims=([-1], [-1])))
        #     tt = torch.stack(tt)
        # else:
        #     tt = torch.tensordot(evaluation, template, dims=([-1], [-1]))
        # tt = torch.transpose(template,dim1=-1,dim0=-2)
        '''
        tt = torch.bmm(template, torch.transpose(evaluation, dim0=-2, dim1=-1))

        tt = torch.softmax(tt, dim=-1)

        # equation (5)
        tt = torch.bmm(torch.transpose(tt, dim0=-2, dim1=-1), template)
        '''
        # !!!!!!!!!!!!!!!!!!!!!!!!!!ERROR
        # pomnoż macierz tt razy wektor template ( kolumnę) aby uzyskać kolejne kolumny ważone
        # re = None
        # if len(template.size()) > 2:
        #     for i in range(template.size(-1)):
        #         if re is None:
        #             re = torch.bmm(tt, torch.unsqueeze(template[:, :, i], dim=-1))
        #         else:
        #             re = torch.cat((re, torch.bmm(tt, torch.unsqueeze(template[:, :, i], dim=-1))), dim=-1)
        # else:
        #     for i in range(template.size(-1)):
        #         if re is None:
        #             re = torch.bmm(tt, torch.unsqueeze(template[:, i], dim=-1))
        #         else:
        #             re = torch.cat((re, torch.bmm(tt, torch.unsqueeze(template[:, i], dim=-1))), dim=-1)
        #
        # tt = re
        '''
        # equation (6)
        tt = torch.abs(torch.sub(tt, evaluation))

        # equation (7), (8)
        attention = torch.softmax(torch.squeeze(self.attention(evaluation)), dim=-1)

        # equation (9)
        # tt = torch.bmm(torch.unsqueeze(attention, -2), tt)
        re = None
        fuk = []
        for j in range(tt.size(-3)):
            for i in range(tt.size(-2)):
                if re is None:
                    re = torch.mul(tt[j, 0], attention[j, 0])
                else:
                    re = torch.add(torch.mul(tt[j, i], attention[j, i]), re)
            fuk.append(re)
            re = None
        tt = torch.stack(fuk)
        # tt = torch.squeeze(tt) stack

        # end of second module

        tt = self.activation(self.hidden(tt))
        tt = self.classifier(tt)

        return torch.softmax(tt, -1)
