This is the code of the paper "Learning to Generate Realistic Noisy Images via Pixel-Level Noise-aware Adversarial Training" in submission ot NeurIPS 2021.

1. Create Environment:

conda env create -f PNGAN.yaml

2. Prepare Dataset:

2.1 Download real denoising benchmarks: SIDD, DND, PolyU, and Nam from their official websites and then put them into the corresponding folders of 'datasets/'.

2.2 Download HD clean image datasets: BSD68, Urban100, Kodak24, DIV2K, and Flickr2K from their official websites and then put them into the corresponding folders of 'datasets/'.

2.3 Use 'prepare_image_data.py' to crop patches at size 256x256 from original noisy and clean images. Put the patches into corresponding folders of 'datasets/'. 'input/' for real noisy images, 'goundtruth/' for clean images. 

2.4 For synthetic setting1, use AWGN to synthesize noisy images, put them into 'S1/'. We also provide script 'Generate_TrainData_HQ_LQ_Denoising_RGB.m' to synthesize AWGN.

2.5 For synthetic setting2, use CycleISP to synthesize noisy images, put them into 'S2/'. Please download the repo of CycleISP from its official website.

3. Prepare Models:

3.1 Download pre-trained denoisers RIDNet, MIRNet, MPRNet from their official websites. Put these pre-trained models into '\pre-trained'. 

3.2 You can also train these denoisers from scratch.

4. Training PNGAN

python3 train_PNGAN.py

5. Using PNGAN to generate fake noisy images. For synthetic setting1, put the fake noisy images into 'S1+PNGAN/'. For synthetic setting2, put the fake noisy images into 'S2+PNGAN/'. 

python3 test_PNGAN.py

6. Use the generated and real noisy images to train or finetune the denoisers. 

python3 train_RIDNet.py or train_MIRNet.py or train_MPRNet.py

You can choose to train or fine-tune by modifying the path and learning rate.

7. Evaluate the performance of denoisers

python3 test_denoiser.py

8. This repo is based on RIDNet, MIRNet, MPRNet, CycleISP. Their official websites are as follows:

8.1 RIDNet: https://github.com/saeed-anwar/RIDNet
8.2 MIRNet: https://github.com/swz30/MIRNet
8.3 MPRNet: https://github.com/swz30/MPRNet
8.4 CycleISP: https://github.com/swz30/CycleISP

We thank their repoes and cite these four works in our paper.

9. LICENSE

According to the regulations, we are required to mention the licenses of the assets used in our paper. Thus, we add the licenses of MIRNet, MPRNet, and CycleISP in this repo. The authors of RIDNet haven't mention their license, yet. Thus, this repo doesn't include the license of RIDNet.