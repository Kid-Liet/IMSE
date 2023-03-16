import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.utils import setup_seed
from utils.deformation import shuffle_remap, affine,non_affine_2d

class ImageDataset(Dataset):
    def __init__(self, root, transforms_, opt):
        self.transforms = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob("%s/T1/*" % root))
        self.files_B = sorted(glob.glob("%s/T2/*" % root))
        self.files_FB = sorted(glob.glob("%s/FT2/*" % root))
        #self.mask = sorted(glob.glob("%s/Label/*" % root))
        
        self.opt = opt
        self.non_affine = non_affine_2d
        self.affine = affine

    def __getitem__(self, index):
        if self.opt['mode'] == "Gen":
           
            item_a = self.transforms(np.load(self.files_FB[index % len(self.files_FB)]).astype(np.float32))    # FB
            item_b = self.transforms(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32)) # RB
        else:
            
            item_a = self.transforms(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))   # RA
            item_b = self.transforms(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))  # RB
        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_a = \
            self.affine(random_numbers=random_numbers, imgs=[item_a], padding_modes=['border'],
                        opt=self.opt)

        random_numbers = torch.rand(8).numpy() * 2 - 1
        item_b= \
            self.affine(random_numbers=random_numbers, imgs=[item_b], padding_modes=['border'],
                        opt=self.opt)
           
    
        # deformation
        item_a = self.non_affine(imgs=[item_a], padding_modes=['border'], opt=self.opt)
        item_b = self.non_affine(imgs=[item_b], padding_modes=['border'], opt=self.opt)
        
     

        return item_a, item_b

    def __len__(self):
        return len(self.files_A)


class TestDataset(Dataset):
    def __init__(self, root, transforms_, opt):
        self.transforms = transforms.Compose(transforms_)
        
        self.files_A = sorted(glob.glob("%s/T1/*" % root))
        self.files_B = sorted(glob.glob("%s/T2/*" % root))
        self.files_FB = sorted(glob.glob("%s/FT2/*" % root))
        
        self.mask = sorted(glob.glob("%s/Label/*" % root))
        self.opt = opt
        self.non_affine = non_affine_2d
        self.affine = affine

        setup_seed(198653)
        self.seeds = torch.randint(18492904, [len(self.files_A)])

    def __getitem__(self, index):
        setup_seed(self.seeds[index % len(self.seeds)])

        # make a seed with numpy generator
        item_MASK = self.transforms(np.load(self.mask[index % len(self.mask)]).astype(np.float32))  # MASK
        item_RA = self.transforms(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))  # RA
        item_RB = self.transforms(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))  # RB
        item_MASK[item_MASK > 0] = 1
        
        item_Mask_A = item_MASK
        item_Mask_B = item_MASK
    
            
            
        if self.opt['mode'] == "Gen":
            
            item_FB = self.transforms(np.load(self.files_FB[index % len(self.files_FB)]).astype(np.float32))
            if self.opt['affine']:
                random_numbers = torch.rand(8).numpy() * 2 - 1
                 item_FB, item_Mask_A = self.affine(random_numbers=random_numbers,
                                                            imgs=[item_FB, item_MASK],
                                                            padding_modes=['border',  'zeros'], opt=self.opt)
                random_numbers = torch.rand(8).numpy() * 2 - 1
                item_RB, item_Mask_B = self.affine(random_numbers=random_numbers,
                                                            imgs=[item_RB, item_MASK],
                                                            padding_modes=['border', 'zeros'], opt=self.opt)

            item_FB, item_Mask_A = self.non_affine(imgs=[item_FB, item_Mask_A],
                                                            padding_modes=['border', 'zeros'], opt=self.opt)
            item_RB, item_Mask_B = self.non_affine(imgs=[item_RB, item_Mask_B],
                                                            padding_modes=['border', , 'zeros'], opt=self.opt)
        else:
            item_FB = torch.zeros((1, 1, 1))
            if self.opt['affine']:
                random_numbers = torch.rand(8).numpy() * 2 - 1
                item_RA, item_Mask_A = self.affine(random_numbers=random_numbers, imgs=[item_RA, item_Mask_A],
                                                   padding_modes=['border', 'zeros'], opt=self.opt)
                random_numbers = torch.rand(8).numpy() * 2 - 1
                item_RB, item_Mask_B = self.affine(random_numbers=random_numbers, imgs=[item_RB, item_Mask_B],
                                                   padding_modes=['border', 'zeros'], opt=self.opt)

            # deformation
            item_RA, item_Mask_A = self.non_affine(imgs=[item_RA, item_Mask_A], padding_modes=['border', 'zeros'],
                                                   opt=self.opt)
            item_RB, item_Mask_B = self.non_affine(imgs=[item_RB, item_Mask_B], padding_modes=['border', 'zeros'],
                                                   opt=self.opt)

        return item_RA, item_RB, item_FB, item_Mask_A, item_Mask_B

    def __len__(self):
        return len(self.files_A)


    
    
    
    
class TestDataset2(Dataset):
    def __init__(self, root, transforms_, opt):
        self.transforms = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob("%s/ct/*" % root))[:100]
        self.files_B = sorted(glob.glob("%s/mr/*" % root))[:100]
        self.A_mask = sorted(glob.glob("%s/ct_label/*" % root))[:100]
        self.B_mask = sorted(glob.glob("%s/mr_label/*" % root))[:100]
        
        self.opt = opt
        self.non_affine = getattr(deformation, 'non_affine_%dd' % opt['dim'])
        self.affine = affine

        setup_seed(198653)
        self.seeds = torch.randint(18492904, [len(self.files_A)])

    def __getitem__(self, index):
        setup_seed(self.seeds[index % len(self.seeds)])

        # make a seed with numpy generator

        item_RA = self.transforms(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))  # RA
        item_RB = self.transforms(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))  # RB
        #rint (np.load(self.A_mask[index % len(self.A_mask)]).astype(np.float32)[:,:,0:1].transpose(2,0,1).shape)
        item_Mask_A = self.transforms(np.load(self.A_mask[index % len(self.A_mask)]).astype(np.float32)[:,:,0]) # MASK
        #print (item_RA.shape,item_Mask_A.shape)
        item_Mask_B = self.transforms(np.load(self.B_mask[index % len(self.B_mask)]).astype(np.float32)[:,:,0]) # MASK

#         item_Mask_A = item_Mask_A[:,0,:,:] +item_Mask_A[:,1,:,:] +item_Mask_A[:,2,:,:]+item_Mask_A[:,3,:,:]
#         item_Mask_A = item_Mask_A.unsqueeze(0)
#         item_Mask_B = item_Mask_B[:,0,:,:] +item_Mask_B[:,1,:,:] +item_Mask_B[:,2,:,:]+item_Mask_B[:,3,:,:]
#         item_Mask_B = item_Mask_B.unsqueeze(0)
       

        if self.opt['mode'] == "Gen":
            
            item_FA = self.transforms(np.load(self.files_FA[index % len(self.files_FA)]).astype(np.float32))
            item_FB = self.transforms(np.load(self.files_FB[index % len(self.files_FB)]).astype(np.float32))
            if self.opt['affine']:
                random_numbers = torch.rand(8).numpy() * 2 - 1
                item_RA, item_FB, item_Mask_A = self.affine(random_numbers=random_numbers,
                                                            imgs=[item_RA, item_FB, item_MASK],
                                                            padding_modes=['border', 'border', 'zeros'], opt=self.opt)
                random_numbers = torch.rand(8).numpy() * 2 - 1
                item_RB, item_FA, item_Mask_B = self.affine(random_numbers=random_numbers,
                                                            imgs=[item_RB, item_FA, item_MASK],
                                                            padding_modes=['border', 'border', 'zeros'], opt=self.opt)

            item_RA, item_FB, item_Mask_A = self.non_affine(imgs=[item_RA, item_FB, item_Mask_A],
                                                            padding_modes=['border', 'border', 'zeros'], opt=self.opt)
            item_RB, item_FA, item_Mask_B = self.non_affine(imgs=[item_RB, item_FA, item_Mask_B],
                                                            padding_modes=['border', 'border', 'zeros'], opt=self.opt)
        else:
            item_FA = torch.zeros((1, 1, 1))
            item_FB = torch.zeros((1, 1, 1))
            if self.opt['affine']:
                random_numbers = torch.rand(8).numpy() * 2 - 1
                item_RA_, item_Mask_A = self.affine(random_numbers=random_numbers, imgs=[item_RA, item_Mask_A],
                                                   padding_modes=['border', 'zeros'], opt=self.opt)
                random_numbers = torch.rand(8).numpy() * 2 - 1
                item_RB_, item_Mask_B = self.affine(random_numbers=random_numbers, imgs=[item_RB, item_RA],
                                                   padding_modes=['border', 'border'], opt=self.opt)

            # deformation
            item_RA_, item_Mask_A = self.non_affine(imgs=[item_RA_, item_Mask_A], padding_modes=['border', 'zeros'],
                                                   opt=self.opt)
            item_RB_, item_Mask_B = self.non_affine(imgs=[item_RB_, item_Mask_B], padding_modes=['border', 'border'],
                                                   opt=self.opt)
        
        
        
        
        #print (item_Mask_A.max(),item_Mask_B.max())
        return item_RA_, item_RB_, item_FA, item_FB, item_Mask_A, item_Mask_B

    def __len__(self):
        return len(self.files_A)
    