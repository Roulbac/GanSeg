import numpy.random as npr
import torch


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Returns images from the buffer.
        Parameters:
            images: the latest generated images from the generator
        Replaces half the minibatch with buffer images
        If buffer is not full, fill buffer and do not replace.
        For a minibatch of size 1, 1/2 chance of replacing image with buffer
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images.detach()
        return_images = []
        n_images = len(images)
        n_storage_images = max(0, min(self.pool_size - self.num_imgs, n_images))
        storage_images = images[:n_storage_images]
        query_images = images[n_storage_images:]
        if self.num_imgs < self.pool_size:
            # Keep inserting
            for image in storage_images:
                image = torch.unsqueeze(image, dim=0)
                self.images.append(image.cpu().clone().detach())  # Store images on CPU
                return_images.append(image)
                self.num_imgs += 1

        minibatch_size = query_images.shape[0]
        if (minibatch_size == 0):
            pass
        elif (minibatch_size == 1):
            p = npr.uniform(0, 1)
            if p < 0.5:
                return_images.append(query_images)
            else:
                index = npr.randint(0, self.num_imgs)
                return_images.append(self.images[index].to(images.device))
                self.images[index] = query_images.cpu().clone().detach()
        else:
            n_drawn_images = min(minibatch_size//2, self.num_imgs)
            buffer_indices = list(range(self.num_imgs))
            images_indices = list(range(minibatch_size))
            npr.shuffle(buffer_indices)
            npr.shuffle(images_indices)
            coupled_indices = zip(
                buffer_indices[:n_drawn_images],
                images_indices[:n_drawn_images]
            )
            for buffer_index, image_index in coupled_indices:
                tmp = self.images[buffer_index]
                self.images[buffer_index] = torch.unsqueeze(
                    query_images[image_index], 0).cpu().clone().detach()
                return_images.append(tmp.to(device=images.device))
            for image_index in images_indices[n_drawn_images:]:
                return_images.append(torch.unsqueeze(
                    query_images[image_index], 0))
        return torch.cat(return_images, 0).detach()
