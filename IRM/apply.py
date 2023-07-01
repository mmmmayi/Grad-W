from torch.utils.data import DataLoader
from Models.dataset import IRMDataset
from Trainer.applier import IRMApplier


PROJECT_DIR = "exp/transCov_sclean_0.001_noLM_encoder_DFLsameF_noWeightmae/wav_18_w"
MODEL_PATH = "exp/transCov_sclean_0.001_noLM_encoder_DFLsameF_noWeightmae/models/model_18.pt"
mode = 'quality'

if __name__ == "__main__":
    ## Dataloader
    '''
    applier_irm_dataset = IRMDataset(
        path="/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/sub_list",
        spk="/data_a11/mayi/project/CAM/lst/val_spk.lst",
        batch_size=1,dur=0,
        sampling_rate=16000, mode=mode)

    applier_loader = DataLoader(
        dataset=applier_irm_dataset,
        batch_size=1,
        shuffle=None,
        num_workers=2)
    '''
    irm_applier = IRMApplier(
        project_dir=PROJECT_DIR,
        test_set_name="BATCH_D",
        sampling_rate=16000,
        model_path=MODEL_PATH,
        applier_dl=None,
        mode = mode)

#    irm_applier.quality_vox1()
    irm_applier.apply()
   
    eval_list = '/data_a11/mayi/project/enhancement/recipes/sitw/xvector/test_list.txt'
    eval_path=['/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/wav',\
    '/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/snr0',\
    '/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/snr5',\
    '/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/snr10',\
    '/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/snr15',\
    '/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/snr20']
    
    for i in range(50,51,2):
        path = "/data_a11/mayi/project/SIP/IRM/exp/transCov_0.001_detachC_noLM_fixed_DFL_mae/models/model_"+str(i)+'.pt'
        print(str(i)+'th epoch')
        irm_applier.speaker_verification(eval_path,eval_list,path)
