%% hyperparameter
clc;clear all;
A = 1;
fc = 2000;
snr_list = [6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 60 100];
p_offset_list = [pi/12 pi/6 pi/4 pi/3 5*pi/12] ;  % Orthogonal offset; I/Q not pi/2
%p_offset = 0

ini_seed1 = randi([0,10000]);
ini_seed2 = randi([0,10000]);

Q_Gain = 1;    %I/Q inbalance: [0 0.5 2]
fc_offset = 0;  %frequency offset
T_delay = 0;   
num_sample = 10000;


%% 16-QAM
m_ary= 16;
mdl = 'QAM';
Label = '16QAM';
dataSetDir = 'D:\RF_ML\data\constellation_data';
set(0, 'DefaultFigureVisible', 'off');

for snr_counter = 1: length(snr_list)
    snr_value = snr_list(snr_counter);
    for p_offset_counter = 1:length(p_offset_list)
        p_offset = p_offset_list(p_offset_counter);
        out = sim(mdl);
        Mod_data = out.QAM(101:end);
        existingFiles = dir(fullfile(dataSetDir, '*.png'));
        
        if ~isempty(existingFiles)
            existingIndices = cellfun(@(x) str2double(regexp(x, '\d+(?=\.png)', 'match')), {existingFiles.name});
            fileCounter = max(existingIndices);
        else
            fileCounter = 0;
        end
        
        num_images = fix(length(Mod_data)/num_sample);
        for i = 1:num_images
            dataSegment = Mod_data((i-1)*num_sample+1:i*num_sample);
            img = figure('visible','off');
            set(img, 'Position', [0 0 224 224]);
            scatterplot(dataSegment);
            axis square; 
            axis off; 
            set(gca, 'position', [0 0 1 1], 'units', 'normalized');
            frame = getframe(gca);
            grayImage = rgb2gray(frame.cdata); 
            grayImage = imresize(grayImage, [224 224]);
            fileName = sprintf('%s_%d_%04d.png', Label,snr_value, fileCounter+i);
            fullPath = fullfile(dataSetDir, fileName);
            imwrite(grayImage, fullPath);    
        end
    end
end
set(0, 'DefaultFigureVisible', 'on');

%% 64-QAM
m_ary= 64;
mdl = 'QAM';
Label = '64QAM';
dataSetDir = 'D:\RF_ML\data\constellation_data';
for snr_counter = 1: length(snr_list)
    snr_value = snr_list(snr_counter);
    for p_offset_counter = 1:length(p_offset_list)
        p_offset = p_offset_list(p_offset_counter);
        out = sim(mdl);
        Mod_data = out.QAM(101:end);
        set(0, 'DefaultFigureVisible', 'off');
        existingFiles = dir(fullfile(dataSetDir, '*.png'));
        if ~isempty(existingFiles)
            existingIndices = cellfun(@(x) str2double(regexp(x, '\d+(?=\.png)', 'match')), {existingFiles.name});
            fileCounter = max(existingIndices);
        else
            fileCounter = 0;
        end
        
        num_images = fix(length(Mod_data)/num_sample);
        for i = 1:num_images
            dataSegment = Mod_data((i-1)*num_sample+1:i*num_sample);
            img = figure('visible','off');
            set(img, 'Position', [0 0 224 224]);
            scatterplot(dataSegment);
            axis square; 
            axis off; 
            set(gca, 'position', [0 0 1 1], 'units', 'normalized');
            frame = getframe(gca);
            grayImage = rgb2gray(frame.cdata); 
            grayImage = imresize(grayImage, [224 224]);
            fileName = sprintf('%s_%d_%04d.png', Label,snr_value, fileCounter+i);
            fullPath = fullfile(dataSetDir, fileName);
            imwrite(grayImage, fullPath);
        end
    end
end
set(0, 'DefaultFigureVisible', 'on');
%% 256-QAM
m_ary= 256;
mdl = 'QAM';
Label = '256QAM';
dataSetDir = 'D:\RF_ML\data\constellation_data';
for snr_counter = 1: length(snr_list)
    snr_value = snr_list(snr_counter);
    for p_offset_counter = 1:length(p_offset_list)
        p_offset = p_offset_list(p_offset_counter);
        out = sim(mdl);
        Mod_data = out.QAM(101:end);
        set(0, 'DefaultFigureVisible', 'off');
        existingFiles = dir(fullfile(dataSetDir, '*.png'));
        if ~isempty(existingFiles)
            existingIndices = cellfun(@(x) str2double(regexp(x, '\d+(?=\.png)', 'match')), {existingFiles.name});
            fileCounter = max(existingIndices);
        else
            fileCounter = 0;
        end
        num_images = fix(length(Mod_data)/num_sample);
        for i = 1:num_images
            dataSegment = Mod_data((i-1)*num_sample+1:i*num_sample);
            img = figure('visible','off');
            set(img, 'Position', [0 0 224 224]);
            scatterplot(dataSegment);
            axis square; 
            axis off; 
            set(gca, 'position', [0 0 1 1], 'units', 'normalized');
            frame = getframe(gca);
            grayImage = rgb2gray(frame.cdata); 
            grayImage = imresize(grayImage, [224 224]);
            fileName = sprintf('%s_%d_%04d.png', Label,snr_value, fileCounter+i);
            fullPath = fullfile(dataSetDir, fileName);
            imwrite(grayImage, fullPath);
        end
    end    
end
set(0, 'DefaultFigureVisible', 'on');
%% 8-PSK
m_ary= 8;
mdl = 'MPSK';
Label = '8PSK';
dataSetDir = 'D:\RF_ML\data\constellation_data';
for snr_counter = 1: length(snr_list)
    snr_value = snr_list(snr_counter);
    out = sim(mdl);
    Mod_data = out.MPSK(101:end);
    set(0, 'DefaultFigureVisible', 'off');
    existingFiles = dir(fullfile(dataSetDir, '*.png'));
    if ~isempty(existingFiles)
        existingIndices = cellfun(@(x) str2double(regexp(x, '\d+(?=\.png)', 'match')), {existingFiles.name});
        fileCounter = max(existingIndices);
    else
        fileCounter = 0;
    end
    
    num_images = fix(length(Mod_data)/num_sample);
    for i = 1:num_images
        dataSegment = Mod_data((i-1)*num_sample+1:i*num_sample);
        img = figure('visible','off');
        set(img, 'Position', [0 0 224 224]);
        scatterplot(dataSegment);
        axis square; 
        axis off; 
        set(gca, 'position', [0 0 1 1], 'units', 'normalized');
        frame = getframe(gca);
        grayImage = rgb2gray(frame.cdata); 
        grayImage = imresize(grayImage, [224 224]);
    
        fileName = sprintf('%s_%d_%04d.png', Label,snr_value, fileCounter+i);
        fullPath = fullfile(dataSetDir, fileName);
        imwrite(grayImage, fullPath);
        
    end
    set(0, 'DefaultFigureVisible', 'on');
end

%% 4-PSK
m_ary= 4;
mdl = 'MPSK';
Label = '4PSK';
dataSetDir = 'D:\RF_ML\data\constellation_data';
for snr_counter = 1: length(snr_list)
    snr_value = snr_list(snr_counter);
    out = sim(mdl);
    Mod_data = out.MPSK(101:end);
    set(0, 'DefaultFigureVisible', 'off');
    existingFiles = dir(fullfile(dataSetDir, '*.png'));
    if ~isempty(existingFiles)
        existingIndices = cellfun(@(x) str2double(regexp(x, '\d+(?=\.png)', 'match')), {existingFiles.name});
        fileCounter = max(existingIndices);
    else
        fileCounter = 0;
    end
    
    num_images = fix(length(Mod_data)/num_sample);
    for i = 1:num_images
        dataSegment = Mod_data((i-1)*num_sample+1:i*num_sample);
        img = figure('visible','off');
        set(img, 'Position', [0 0 224 224]);
        scatterplot(dataSegment);
        axis square; 
        axis off; 
        set(gca, 'position', [0 0 1 1], 'units', 'normalized');
        frame = getframe(gca);
        grayImage = rgb2gray(frame.cdata); 
        grayImage = imresize(grayImage, [224 224]);
    
        fileName = sprintf('%s_%d_%04d.png', Label,snr_value, fileCounter+i);
        fullPath = fullfile(dataSetDir, fileName);
        imwrite(grayImage, fullPath);
        
    end
    set(0, 'DefaultFigureVisible', 'on');
end

%% 2-PSK
m_ary= 2;
mdl = 'MPSK';
Label = 'BPSK';
dataSetDir = 'D:\RF_ML\data\constellation_data';
for snr_counter = 1: length(snr_list)
    snr_value = snr_list(snr_counter);
    out = sim(mdl);
    Mod_data = out.MPSK(101:end);
    set(0, 'DefaultFigureVisible', 'off');
    existingFiles = dir(fullfile(dataSetDir, '*.png'));
    if ~isempty(existingFiles)
        existingIndices = cellfun(@(x) str2double(regexp(x, '\d+(?=\.png)', 'match')), {existingFiles.name});
        fileCounter = max(existingIndices);
    else
        fileCounter = 0;
    end
    
    num_images = fix(length(Mod_data)/num_sample);
    for i = 1:num_images
        dataSegment = Mod_data((i-1)*num_sample+1:i*num_sample);
        img = figure('visible','off');
        set(img, 'Position', [0 0 224 224]);
        scatterplot(dataSegment);
        axis square; 
        axis off; 
        set(gca, 'position', [0 0 1 1], 'units', 'normalized');
        frame = getframe(gca);
        grayImage = rgb2gray(frame.cdata); 
        grayImage = imresize(grayImage, [224 224]);
    
        fileName = sprintf('%s_%d_%04d.png', Label,snr_value, fileCounter+i);
        fullPath = fullfile(dataSetDir, fileName);
        imwrite(grayImage, fullPath);
        
    end
    set(0, 'DefaultFigureVisible', 'on');
end
