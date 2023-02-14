from torch.utils.data import DataLoader
from Models.dataset import IRMDataset
from Trainer.applier import IRMApplier


PROJECT_DIR = "exp/transCov_twin_8s_lr0.3_w0.95_hn0.92_n0.001_s8_th0.05_L_in_all/wav_4"
MODEL_PATH = "exp/transCov_twin_8s_lr0.3_w0.95_hn0.92_n0.001_s8_th0.05_L_in_all/models/model_4.pt"
mode = 'quality'

if __name__ == "__main__":
    ## Dataloader
    applier_irm_dataset = IRMDataset(
        path="/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/sub_list",
        spk="/data_a11/mayi/project/CAM/lst/val_spk.lst",
        batch_size=1,dur=0,
        sampling_rate=16000, mode=mode, max_size=10000, data=mode)

    applier_loader = DataLoader(
        dataset=applier_irm_dataset,
        batch_size=1,
        shuffle=None,
        num_workers=2)

    irm_applier = IRMApplier(
        project_dir=PROJECT_DIR,
        test_set_name="BATCH_D",
        sampling_rate=16000,
        model_path=MODEL_PATH,
        applier_dl=applier_loader,
        mode = mode)
#    irm_applier.restore()
#    irm_applier.quality_vox1()
    irm_applier.apply()
