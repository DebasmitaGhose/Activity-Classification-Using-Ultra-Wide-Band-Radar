clear all; 
close all; 
clc

%% Query user for logfile
[fnm,dnm] = uigetfile('*.csv');
fprintf('Reading logfile %s\n',fullfile(dnm,fnm));
[cfg,req,scn,det] = readMrmRetLog(fullfile(dnm,fnm));

 %% Pull out the raw scans (if saved)
rawscansI = find([scn.Nfilt] == 1);
rawscansV = reshape([scn(rawscansI).scn],[],length(rawscansI))';

%% Create the waterfall horizontal (also know as range or fast time) and vertical axes (also known as scan number or slow time)
Tbin = 32/(512*1.024);  % ns   (T2-T1)/N=Tbin  from the formulas in the biosensors paper
T0 = 0; % ns
c = 0.29979;  % m/ns
Rbin = c*(Tbin*(0:size(rawscansV,2)-1) - T0)/2;  % Range Bins in meters
IDdat = 1:size(rawscansV,1);

%% Plot raw data as a waterfall
figure;
imagesc(Rbin,IDdat,rawscansV);
xlabel('Range (m)')
ylabel('Scan Number')
title('Waterfall plot of raw scans')
colorbar
drawnow

%% bandpass filter
[b,a] = filt_coefs;
bprawscansV = filter(b,a,rawscansV,[],2);

figure; % Visualizing the raw bandpass data
imagesc(Rbin,IDdat,bprawscansV);
xlabel('Range (m)')
ylabel('Scan Number')
title('Waterfall plot of bandpass filtered scans')
colorbar
drawnow

%% filtering to remove clutter
% We apply the following differece equation to remove static clutter
b = [1 -0.6 -0.3 -0.1]; 
a = 1;

figure; freqz(b,a); %FIR4 highpass 0.245Hz:cutoff freq, 8Hz=Fs 

noclutterscansV = filter(b,a,bprawscansV,[],1); %filter each column % This we we got rid of static clutter and also preserve motion information (note we might have lost vital sign information though)

figure; % Visualizing the clutter removed data
imagesc(Rbin,IDdat,noclutterscansV);
xlabel('Range (m)')
ylabel('Scan Number')
title('Waterfall plot of bandpass filtered scans')
colorbar
drawnow

%% find envelope of the nonclutter rawscans

[b,a] = butter(6,0.4);  
%freqz(b,a) % This is a low pass filter.
envNoClutterscansV = filter(b,a,abs(noclutterscansV),[],2);% envelope of each scan or each row of the noclutterscansV
envNoClutterscansV = max(envNoClutterscansV,0);% now just ignoring anything less than zero

figure;
imagesc(Rbin,IDdat,envNoClutterscansV);
xlabel('Range (m)')
ylabel('Scan Number')
title('Waterfall plot of envelope of no clutter scan')
colorbar
drawnow

%% Range estimation and localization

%ToDo 1: Write the code to localize the human body. Plot scan number (in x-axis) vs range/distance to human body (in y-axis).
weighted_matrix = envNoClutterscansV.*Rbin;
P = [];
for i=1:size(envNoClutterscansV,2)
    [peaks_1,ind_1] = findpeaks(envNoClutterscansV(:,i));
    %P = [P;ind_1];
end
%plot(Rbin(ind_1),IDdat);
for i = 1:size(envNoClutterscansV,1)
    [~,ind] = max(envNoClutterscansV(i,:));
    P = [P;ind];
end
[p,a]= findpeaks(Rbin(P));
pre = 1:min(a)-1;
pre = size(pre,2);
post = max(a)+1:size(envNoClutterscansV,1);
post = size(post,2);
plot([zeros(pre,1)' IDdat(a(1):a(end)) zeros(post,1)'],Rbin(P))
figure;
plot(IDdat,Rbin(P));

for j = 1:size(envNoClutterscansV,1)
    [peaks_2,ind_2] = findpeaks(envNoClutterscansV(j,:));
end

%peaks = medfilt1(peaks);
disp(size(peaks_1));
disp(size(peaks_2));
[~,idx] = findpeaks(peaks_1,'Threshold',20);%peaks where succesive values drop by x
figure;
localized_range = Rbin(idx);
pre_range = 1:(min(idx)-1);
pre_range = size(pre_range,2);
post_range = (max(idx)+1):size(envNoClutterscansV,2);
post_range = size(post_range,2);
range_total = [zeros(pre_range,1)' Rbin(idx(1):idx(end)) zeros(post_range,1)'];
figure;
%plot(IDdat,range_total);
envNoClutterscansV(:,1:min(idx)-1) = 0;
envNoClutterscansV(:,max(idx)+1:size(envNoClutterscansV,2)) = 0;
th_scans = envNoClutterscansV(:,[1:min(idx)-1,idx(1):idx(end),(max(idx)+1):size(envNoClutterscansV,2)]);
imagesc(Rbin(idx(1):idx(end)),IDdat,th_scans);
xlabel('Range (m)')
ylabel('Scan Number')
title('Waterfall plot of envelope of no clutter scan_localized')
colorbar
drawnow


%% Activity Recognition

%ToDo 2: Extract relevant features and train a contactless activity classifier.
th_win = [];
th_scans_win_first = th_scans(1:25,:);
th_win = [th_win,th_scans_win_first];
mean_arr = [];
max_arr = [];
min_arr = [];
fft_mean_arr = [];
fft_max_arr = [];
fft_min_arr = [];
fft_var_arr = [];
fft_std_arr = [];
fft_med_arr = [];
count = 0;
for k = 1:20:(size(th_scans,1))-25
    
    if(k+25<size(th_scans,1)-25)
        
        
        %k=k-5;
        th_scans_win = (th_scans(k:k+24,:));
        
        disp(k);
        disp(k+24);
        mean_win = mean(th_scans_win,1);
        max_win = max(th_scans_win);
        min_win = min(th_scans_win);
        fft_mean_win = mean(fft(th_scans_win,[],1),1);
        fft_max_win = max(fft(th_scans_win,[],1));
        fft_min_win = min(fft(th_scans_win,[],1));
        fft_var_win = var(fft(th_scans_win,[],1),[],1);
        fft_std_win = std(fft(th_scans_win,[],1),[],1);
        fft_med_win = median(fft(th_scans_win,[],1),1);
        %disp(size(mean_win));  
        count = count+1;
        %mean_arr(k,:) = mean_win;
        %max_arr(k,:) = max_win;
        %min_arr(k,:) = min_win;
        %fft_mean_arr(k,:) = fft_mean_win;
        %fft_max_arr(k,:) = fft_max_win;
        %fft_min_arr(k,:) = fft_min_win;
        %fft_var_arr(k,:) = fft_var_win;
        %fft_std_arr(k,:) = fft_std_win;
        %fft_med_arr(k,:) = fft_med_win;
    else
        break;
    end
        mean_arr = [mean_arr;mean_win];
        max_arr = [max_arr; max_win];
        min_arr = [min_arr; min_win];
        fft_mean_arr = [fft_mean_arr;fft_mean_win];
        fft_max_arr = [fft_max_arr;fft_max_win];
        fft_min_arr = [fft_min_arr;fft_min_win];
        fft_var_arr = [fft_var_arr;fft_var_win];
        fft_std_arr = [fft_std_arr;fft_std_win];
        fft_med_arr = [fft_med_arr;fft_med_win];    
    %k=k-5;
    
end
mean_arr = reshape(mean_arr,size(mean_arr,1)*size(mean_arr,2),1);
max_arr = reshape(max_arr,size(max_arr,1)*size(max_arr,2),1);
min_arr = reshape(min_arr,size(min_arr,1)*size(min_arr,2),1);
fft_mean_arr = reshape(fft_mean_arr,size(fft_mean_arr,1)*size(fft_mean_arr,2),1);
fft_min_arr = reshape(fft_min_arr,size(fft_min_arr,1)*size(fft_min_arr,2),1);
fft_max_arr = reshape(fft_max_arr,size(fft_max_arr,1)*size(fft_max_arr,2),1);
fft_var_arr = reshape(fft_var_arr,size(fft_var_arr,1)*size(fft_var_arr,2),1);
fft_std_arr = reshape(fft_std_arr,size(fft_std_arr,1)*size(fft_std_arr,2),1);
fft_med_arr = reshape(fft_med_arr,size(fft_med_arr,1)*size(fft_med_arr,2),1);

disp(size(mean_arr));        
%disp(size(th_scans_win));
disp(size(fft_min_arr));
disp(size(min_arr));
disp(count); 

%stacking
feature_vector = cat(2,mean_arr,max_arr,min_arr,fft_mean_arr,fft_max_arr,fft_min_arr,fft_var_arr,fft_std_arr,fft_med_arr);

