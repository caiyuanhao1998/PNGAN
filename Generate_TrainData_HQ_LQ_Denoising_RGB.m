function Generate_TrainData_HQ_LQ_Denoising_RGB()
% generate training data for gray denoising
clear all; close all; clc
path_original = 'C:\polyu';
%dataset = {'CBSD68', 'DNSet9', 'Kodak24', 'Set5', 'Set14', 'BSD100', ...
%              'Urban100', 'Manga109', 'DIV2K40', 'LIVE1'};
dataset = {'clean'};
%dataset  = {'Set5', 'Set14C', 'B100', 'Urban100', 'Manga109'};
ext = {'*.jpg', '*.png', '*.bmp'};
% sigmaAll = [10, 30, 50, 70];
sigmaAll = [50];

for idx_set = 1:length(dataset)
    fprintf('Processing %s:\n', dataset{idx_set});
    filepaths = [];
    for idx_ext = 1:length(ext)
        filepaths = cat(1, filepaths, dir(fullfile(path_original, dataset{idx_set}, ext{idx_ext})));
    end
    for idx_im = 1:length(filepaths)
        name_im = filepaths(idx_im).name;
        fprintf('%d. %s: ', idx_im, name_im);
        imgori = imread(fullfile(path_original, dataset{idx_set}, name_im));
%         % convert RGB to gray
%         if size(imgori, 3) == 3
%             imgori_gray = rgb2gray(imgori);
%         else
%             imgori_gray = imgori;
%         end
        % save HQ gray image
%         folder_HQ = fullfile('./DIVFlickr2K/DIVFlickr2K_HQ');
%         if ~exist(folder_HQ)
%             mkdir(folder_HQ)
%         end
%         fn_HQ = fullfile(folder_HQ, name_im);
%         imwrite(imgori, fn_HQ, 'png');
        
        
        label = im2double(imgori);
        
        for IdxSigma = 1:length(sigmaAll)
            %randn('seed',0); % for training, don't set 'seed'
            sigma = sigmaAll(IdxSigma);
            fprintf('sigma%d ', sigma);
            input = single(label + sigma/255*randn(size(label)));
            input = im2uint8(input);
            % folder
            folder_LQ = fullfile('C:\Users\caiyuanhao\Desktop\presentation_nips\polyu', ['X', num2str(sigma)]);
            %folder_HQ = fullfile('./DIVFlickr2K/DIVFlickr2K_HQ', ['N', num2str(sigma)]);
            
            if ~exist(folder_LQ)
                mkdir(folder_LQ)
            end
%             if ~exist(folder_HQ)
%                 mkdir(folder_HQ)
%             end

            fn_LQ = fullfile(folder_LQ, [name_im(1:end-4), '.png']);
            %fn_HQ = fullfile(folder_HQ, [name_im(1:end-4), '_HQ_N', num2str(sigma), '.png']);
            
            imwrite(input, fn_LQ);
            %imwrite(imgori_gray, fn_HQ, 'png');

        end
        fprintf('\n');
    end
    fprintf('\n');
end
end
function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1
    sz = size(imgs);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2),:);
end
end

function ImLR = imresize_noise(ImHR, scale, sigma)
% ImLR and ImHR are uint8 data
% downsample by Bicubic
ImDown = imresize(ImHR, 1/scale, 'bicubic'); % 0-255
ImDown = single(ImDown); % 0-255
ImDownNoise = ImDown + single(sigma*randn(size(ImDown))); % 0-255
ImLR = uint8(ImDownNoise); % 0-255
end
