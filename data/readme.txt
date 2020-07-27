The folder random_remove contains the corrupted skeleton for training the denoising autoencoder based on SeBiReNet.

In the subfolders, "remove1" means only one joint is randomly selected and moved to the invalid position. Likewise, we can under stand other folder names. train_gt.json and test_gt.json are clear version corresponding to the corrupted train.json and test.json files.

dataPath = {
    'train': ['./remove1/train/APE_train.json',              			  './remove2/train/APE_train.json',
              './remove3/train/APE_train.json',
              './remove4/train/APE_train.json',
              './remove5/train/APE_train.json']
'test': ['./remove1/test/APE_test.json',
         './remove2/test/APE_test.json',
         './remove3/test/APE_test.json',
         './remove4/test/APE_test.json',
         './remove5/test/APE_test.json']
'train_gt': ['./remove1/train/APE_train_gt.json',
             './remove1/train/APE_train_gt.json',
             './remove3/train/APE_train_gt.json',
             './remove4/train/APE_train_gt.json',
             './remove5/train/APE_train_gt.json']
'test_gt': ['./remove1/test/APE_test_gt.json',
            './remove2/test/APE_test_gt.json',
            './remove3/test/APE_test_gt.json',
            './remove4/test/APE_test_gt.json',
            './remove5/test/APE_test_gt.json']}

This Dataset is created and thanks to one of my former colleague Yun Zhong.



