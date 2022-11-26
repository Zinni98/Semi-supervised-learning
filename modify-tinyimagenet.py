import os;
if __name__ == "__main__":
    data_dir = "data/tiny-imagenet-200/"
    # loop over all lines in val_annotations.txt and save in a dictionary with class as key and list of images as value
    val_annotations = open("data/tiny-imagenet-200/val/val_annotations.txt", "r")
    val_annotations_dict = {}
    for line in val_annotations:
        line = line.split("\t")
        if line[1] not in val_annotations_dict:
            val_annotations_dict[line[1]] = []
        val_annotations_dict[line[1]].append(line[0])
    val_annotations.close()

    print(val_annotations_dict["n01443537"])
    print(len(val_annotations_dict))

    # print number of folders in train folder
    print(len(os.listdir(data_dir + "train/")))
    
    # #create a new folder val1 in data/tiny-imagenet-200/
    # os.mkdir(data_dir + "val1")
    # #create a new folder for each class in val1
    # for key in val_annotations_dict:
    #     os.mkdir(data_dir + "val1/" + key)

    # #make a copy of all images from val to val1 according to their class
    # for key in val_annotations_dict:
    #     for image in val_annotations_dict[key]:
    #         os.system("cp " + data_dir + "val/images/" + image + " " + data_dir + "val1/" + key + "/")