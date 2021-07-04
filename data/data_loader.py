import torch
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import datetime
from pandas import read_csv, concat
from scipy.spatial.transform import Rotation as R
import numpy as np
from PIL import Image
from torchvision import transforms


# data_loader.py: Generates custom PyTorch datasets for solar energy generation prediction

def show_im(image):
    # To plot Torch Tensor Images
    plt.imshow(image.permute(1, 2, 0))
    plt.show()


def calc_dataset_std(dataset):
    # Note: this assumes means are already subtracted
    ir_std = 0
    power_std = 0
    image_stds = torch.FloatTensor([0, 0, 0])
    sun_stds = torch.zeros(dataset[0]["sun pose"].type(torch.FloatTensor).shape)
    n = 0

    for data in dataset:
        ir_std += data["ir"].type(torch.FloatTensor) ** 2
        power_std += data["power"].type(torch.FloatTensor) ** 2
        image = data["image"].type(torch.FloatTensor)
        image_stds[0] += torch.mul(image[0], image[0]).mean()
        image_stds[1] += torch.mul(image[1], image[1]).mean()
        image_stds[2] += torch.mul(image[2], image[2]).mean()
        pose = data["sun pose"].type(torch.FloatTensor)
        sun_stds += torch.mul(pose, pose)
        n += 1
        if n % 100 == 0:
            print(n)

    return torch.sqrt(torch.div(ir_std, (n - 1))), torch.sqrt(torch.div(power_std, (n - 1))), \
           torch.sqrt(torch.div(image_stds, (n - 1))), torch.sqrt(torch.div(sun_stds, (n - 1)))


def calc_dataset_means(dataset):
    ir_mean = torch.zeros(dataset[0]["ir"].type(torch.FloatTensor).shape)
    power_mean = torch.zeros(dataset[0]["power"].type(torch.FloatTensor).shape)
    image_mean = torch.zeros(dataset[0]["image"].type(torch.FloatTensor).shape)
    sun_mean = torch.zeros(dataset[0]["sun pose"].type(torch.FloatTensor).shape)
    n = 0
    for data in dataset:
        power_mean += data["power"].type(torch.FloatTensor)
        ir_mean += data["ir"].type(torch.FloatTensor)
        image_mean += data["image"].type(torch.FloatTensor)
        sun_mean += data["sun pose"].type(torch.FloatTensor)
        n += 1
        if n % 100 == 0:
            print(n)
    channel_mean = [0, 0, 0]
    for i in range(3):
        channel_mean[i] = image_mean[i, :, :].mean() / n

    return ir_mean / n, power_mean / n, channel_mean, sun_mean / n


def convert_date_string_to_unix_seconds(date_and_time):
    #  Convert a date & time to a unix timestamp in seconds.

    # Extract microseconds part and convert it to seconds
    microseconds_str = date_and_time.split('_')[-1]
    seconds_remainder = float('0.' + microseconds_str)

    # Extract date without microseconds and convert to unix timestamp
    # The added 'GMT-0400' indicates that the provided date is in the
    # Toronto (eastern) timezone during daylight saving
    date_to_sec_str = date_and_time.replace('_' + microseconds_str, '')
    seconds_to_sec = datetime.datetime.strptime(date_to_sec_str + ' GMT-0400', "%Y_%m_%d_%H_%M_%S GMT%z").timestamp()

    # Add the microseconds remainder
    return seconds_to_sec + seconds_remainder


class SolarPredDataset(Dataset):
    # Creates dataset for mobile robot solar energy generation prediction from the The Canadian Planetary Emulation
    # Terrain Energy-Aware Rover Navigation Dataset

    def __init__(self, input_dir):
        # Load Irradiance Data
        frame1 = read_csv(os.path.join(input_dir, "run1", "pyranometer.txt"))
        frame2 = read_csv(os.path.join(input_dir, "run2", "pyranometer.txt"))
        frame3 = read_csv(os.path.join(input_dir, "run3", "pyranometer.txt"))
        frame4 = read_csv(os.path.join(input_dir, "run4", "pyranometer.txt"))
        frame5 = read_csv(os.path.join(input_dir, "run5", "pyranometer.txt"))
        frame6 = read_csv(os.path.join(input_dir, "run6", "pyranometer.txt"))
        self.ir = concat([frame1, frame2, frame3, frame4, frame5, frame6], ignore_index=True)

        # Load Sun Position Data
        frame1 = read_csv(os.path.join(input_dir, "run1", "relative-sun-position.txt"))
        frame2 = read_csv(os.path.join(input_dir, "run2", "relative-sun-position.txt"))
        frame3 = read_csv(os.path.join(input_dir, "run3", "relative-sun-position.txt"))
        frame4 = read_csv(os.path.join(input_dir, "run4", "relative-sun-position.txt"))
        frame5 = read_csv(os.path.join(input_dir, "run5", "relative-sun-position.txt"))
        frame6 = read_csv(os.path.join(input_dir, "run6", "relative-sun-position.txt"))
        self.sun_pos = concat([frame1, frame2, frame3, frame4, frame5, frame6], ignore_index=True)

        # Load list of image files
        self.image_file_list = [os.path.join(input_dir, "run1", "omni_stitched_image", path) for path in
                                os.listdir(os.path.join(input_dir, "run1", "omni_stitched_image"))] + [
                                   os.path.join(input_dir, "run2", "omni_stitched_image", path) for path in
                                   os.listdir(os.path.join(input_dir, "run2", "omni_stitched_image"))] + [
                                   os.path.join(input_dir, "run3", "omni_stitched_image", path) for path in
                                   os.listdir(os.path.join(input_dir, "run3", "omni_stitched_image"))] + [
                                   os.path.join(input_dir, "run4", "omni_stitched_image", path) for path in
                                   os.listdir(os.path.join(input_dir, "run4", "omni_stitched_image"))] + [
                                   os.path.join(input_dir, "run5", "omni_stitched_image", path) for path in
                                   os.listdir(os.path.join(input_dir, "run5", "omni_stitched_image"))] + [
                                   os.path.join(input_dir, "run6", "omni_stitched_image", path) for path in
                                   os.listdir(os.path.join(input_dir, "run6", "omni_stitched_image"))]

        # Store previously calculated dataset means and standard deviations
        self.ir_mean = 123.9104
        self.power_mean = 249.1758
        self.ir_std = 77.4094
        self.power_std = 291.4246
        self.solar_efficiency = 1
        self.sun_stds = torch.FloatTensor([0.1147, 0.1024, 0.7396, 0.6553])
        self.sun_means = torch.FloatTensor([629.2003 / 14271, -1066.0398 / 14271, 4276.5586 / 14271, 6782.6909 / 14271])

        # Perform transformations on input images: downsample, transform to tensor, and normalize
        self.image_means = [97.7848, 122.5085, 124.8840]
        self.image_stds = [68.4292, 83.8599, 88.3367]
        self.image_transform = transforms.Compose([transforms.Resize([240, 1160]), transforms.ToTensor(),
                                                   transforms.Normalize(mean=self.image_means, std=self.image_stds)])

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        image_path = self.image_file_list[idx]  # Sample image file path
        # Interpolate telemetry using image file timestamp
        time_stamp = convert_date_string_to_unix_seconds(image_path.split("/")[-1][12:-4])
        row = self.ir.iloc[(self.ir["time"] - time_stamp).abs().argsort()[0]]
        ir_val = torch.FloatTensor(np.asarray([row[1]]))

        row = self.sun_pos.iloc[(self.ir["time"] - time_stamp).abs().argsort()[0]]
        sun_orientation = torch.FloatTensor(row[4:].values.tolist())

        # Load and transform input image
        image = Image.open(image_path)
        image = self.image_transform(image)

        # Calculate power generated, simplified geometric model for effective surface area
        r = R.from_quat(sun_orientation)
        sx = r.as_matrix()[0, 2]
        sy = r.as_matrix()[1, 2]
        area = 9 * max(sx, -sx) + 3 * max(0, -sy)
        power = torch.FloatTensor([area * ir_val * self.solar_efficiency])

        sample = {'image': image, 'ir': ir_val, "sun pose": sun_orientation, "power": power}
        return sample


if __name__ == "__main__":
    input_dir = "data"
    dataset = SolarPredDataset(input_dir)
    show_im(dataset[42]["image"])
    print(len(dataset))
