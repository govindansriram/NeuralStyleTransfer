import numpy as np
from PIL import Image
from MiscMethods import image_loader, show_image
import torch
import torch.optim as optim
from ModVGG import ModifiedVGG
from torchvision.utils import save_image


def content_cost_function(content_vector,
                          generated_vector):
    cost = torch.mean((content_vector - generated_vector) ** 2)

    return cost


def gram_matrix(vector):
    dim_1, dim_2, dim_3, dim_4 = vector.size()
    vector = vector.view(dim_1 * dim_2, dim_3 * dim_4)
    gram = torch.mm(vector, vector.t())

    return gram / (dim_1 * dim_2 * dim_3 * dim_4)


def style_cost_function(style_vector,
                        generated_vector):
    gram_style = gram_matrix(style_vector)
    gram_gen = gram_matrix(generated_vector)

    cost = torch.mean((gram_style - gram_gen) ** 2)

    return cost


class BloumModel:

    def __init__(self,
                 style_image_path,
                 content_image_path,
                 image_size,
                 alpha,
                 beta,
                 learning_rate,
                 epochs,
                 device="cpu"):

        self.__model = ModifiedVGG(device)
        self.__model.eval()
        self.__style_image = image_loader(image_size,
                                          image_path=style_image_path,
                                          device=device)

        self.__content_image = image_loader(image_size,
                                            image_path=content_image_path,
                                            device=device)

        # generated_image = np.random.randint(low=0,
        #                                     high=255,
        #                                     size=(image_size, image_size))

        # generated_image = Image.fromarray(generated_image.astype('uint8')).convert(mode='RGB')

        # self.__generated_image = image_loader(image_size,
        #                                       picture=generated_image,
        #                                       device=device).requires_grad_(True)

        self.__generated_image = self.__content_image.clone().requires_grad_(True)

        self.__alpha = alpha
        self.__beta = beta
        self.__epochs = epochs

        self.__optimizer = optim.Adam([self.__generated_image], lr=learning_rate)

    def get_cost(self,
                 style_vector_list,
                 generated_vector_list_cont,
                 generated_vector_list_style,
                 content_vector_list,
                 lmbda):

        content_cost = 0
        style_cost = 0

        for content, gen_cont in zip(content_vector_list, generated_vector_list_cont):
            content_cost += content_cost_function(content, gen_cont)

        for style, gen_style in zip(style_vector_list, generated_vector_list_style):
            style_cost += content_cost_function(style, gen_style)

        content_cost *= self.__alpha
        style_cost *= (lmbda * self.__beta)

        return content_cost + style_cost

    def train_model(self):

        for epoch in range(self.__epochs):
            print(epoch)
            self.__optimizer.zero_grad()

            # gen_styl_vec_list = self.__model(self.__generated_image, [1, 2, 3, 4, 5])
            # gen_con_vec_list = self.__model(self.__generated_image, [4])
            # con_vec_list = self.__model(self.__content_image, [4])
            # gen_con_vec_list = self.__model(self.__generated_image, [1, 2, 3, 4, 5])
            # con_vec_list = self.__model(self.__content_image, [1, 2, 3, 4, 5])
            # styl_vec_list = self.__model(self.__style_image, [1, 2, 3, 4, 5])

            gen_styl_vec_list = self.__model(self.__generated_image, [0, 5, 10, 19, 28])
            gen_con_vec_list = self.__model(self.__generated_image, [21])
            con_vec_list = self.__model(self.__content_image, [21])
            styl_vec_list = self.__model(self.__style_image, [0, 5, 10, 19, 28])

            loss = self.get_cost(styl_vec_list,
                                 gen_con_vec_list,
                                 gen_styl_vec_list,
                                 con_vec_list,
                                 1)

            loss.backward()
            self.__optimizer.step()

            if epoch % 50 == 0:
                save_image(self.__generated_image, "./Data/GenIm.png")

                print(loss.item())

        return self.__generated_image

    def get_generated_image(self):
        return self.__generated_image

    def get_style_image(self):
        return self.__style_image

    def get_content_image(self):
        return self.__content_image

    def get_model(self):
        return self.__model


if __name__ == '__main__':
    path_style = "./Data/StyleStarryNight.jpeg"
    path_content = "./Data/StanfordContent.jpeg"

    bloum = BloumModel(path_style,
                       path_content,
                       512,
                       1e-1,
                       1e9,
                       0.01,
                       7000)

    model = ModifiedVGG("cpu")
    show_image(bloum.train_model(), "final product")
