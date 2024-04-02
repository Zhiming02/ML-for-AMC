clc;clear all;
A = 1;
fc = 2000;
num_sample = 1000;
num_images = 5;
ini_seed1 = 2002;
ini_seed2 =628;
T_stop = num_images +0.1;

snr_list = [10 15 20 25 30];

img_p_offset_list = [0 15 30 45];
p_offset_list = [0 pi/6 pi/4 pi/3] ;  % Orthogonal offset; I/Q not pi/2
Q_Gain_list = [1 0.5 2];    %I/Q inbalance: [0 0.5 2]
T_delay_list = [0 1 2]; 
fc_offset_list = [0];  %frequency offset
m_ary= 8;
mdl = 'Sim_Model';
datasetDir = 'D:\RF_ML\data\1';
Label = '8PAM';  


%% 16-QAM
set_param(mdl, 'StopTime', num2str(T_stop));
%dataSetDir = 'D:\RF_ML\data\dataset_2';
set(0, 'DefaultFigureVisible', 'off');
%run_time = 0;
existingFiles = dir(fullfile(dataSetDir, '*.png'));
delete(gcp('nocreate'));
parpool(5);

for snr_counter = 1: length(snr_list)
    snr_value = snr_list(snr_counter);
    for p_offset_counter = 1:length(p_offset_list)
        p_offset = p_offset_list(p_offset_counter);
        for Q_Gain_counter = 1:length(Q_Gain_list)
        Q_Gain = Q_Gain_list(Q_Gain_counter);
            for img_p_offset_counter = 1:length(img_p_offset_list)
            img_p_offset = img_p_offset_list(img_p_offset_counter);
                for T_delay_counter = 1:length(T_delay_list)
                T_delay = T_delay_list(T_delay_counter);
                    for fc_offset_counter = 1:length(fc_offset_list)
                    fc_offset = fc_offset_list(fc_offset_counter);
                    %tic;
                    Mod_data = zeros(1, num_sample*num_images);
                    out = sim(mdl);
                    Mod_data = out.QAM(101:end);
                    SaveImgfcn(Mod_data, num_sample, num_images,Label, snr_counter, p_offset_counter,Q_Gain_counter, img_p_offset_counter, T_delay,fc_offset_counter,datasetDir);
                    clear out Mod_data existingFiles existingIndices
                    %run_time = toc;  
                    %fprintf('Time taken for one loop iteration: %f seconds.\n', run_time);
                    end
                end     
            end
        end
    end
end

delete(gcp('nocreate'));
set(0, 'DefaultFigureVisible', 'on');
clc;clear all;
