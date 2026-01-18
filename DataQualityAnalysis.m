%% Quality Analysis
clear
myFolder = 'C:\Users\Condell\Documents\data\Curated_Data\Long-Term_tES' %Change folder for each dataset (formatted for raw dataset)
filePattern = fullfile(myFolder, '**/*.snirf');
theFiles = dir(filePattern);
subjcount=1;
prevsubj=[];
subjSNR=[];
subjMR=[];
SNRall=[];
MRall=[];
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    %fprintf(1, 'Now reading %s\n', fullFileName);
    currentfile=SnirfLoad(fullFileName);
    currentdata=(currentfile.data.dataTimeSeries);
    ChannelSNR=20*log10(mean(currentdata)./std(currentdata));
    currentOD = hmrR_Intensity2OD(currentfile.data);
    [tInc, tIncCh]=hmrR_MotionArtifactByChannel(currentOD,currentfile.probe, [], [], [], 0.5, 0.5, 20, 5);
    ChannelMR=1-sum(tIncCh{1})./length(tIncCh{1});

    [filepath, filename]=fileparts(fullFileName);
    [~, subj]=fileparts(filepath);
    if k==1 || strcmp(prevsubj,subj)==1
        subjSNR(end+1,:)=ChannelSNR;
        subjMR(end+1,:)=ChannelMR;
    else
        SubjectSNR.(prevsubj)=subjSNR;
        SubjectMR.(prevsubj)=subjMR;
        subjcount=subjcount+1;
        subjSNR=ChannelSNR;
        subjMR=ChannelMR;
        disp(baseFileName)
    end
     if k==length(theFiles)
       SubjectSNR.(subj)=subjSNR;
       SubjectMR.(subj)=subjMR;  
     end
%     if k==length(theFiles)
%       SubjectSNR.(prevsubj)=subjSNR;
%       SubjectMR.(prevsubj)=subjMR;  
%    end
    prevsubj=subj;
    SNRall(end+1,:)=ChannelSNR;
    MRall(end+1,:)=ChannelMR;
end

%% Manually Reordering Subjects for Expert-Novice ONLY
SubjNames=fieldnames(SubjectSNR);
OrderedSubjNames={SubjNames{9};SubjNames{10};SubjNames{11};SubjNames{12};SubjNames{13};SubjNames{14};
                    SubjNames{6};SubjNames{7};SubjNames{8};SubjNames{3};SubjNames{4};SubjNames{5};SubjNames{1};SubjNames{2}};
OrderedSubjectSNR=orderfields(SubjectSNR,OrderedSubjNames);
OrderedSubjectMR=orderfields(SubjectMR,OrderedSubjNames);
%% Not Actually Reordering Subjects for Learning Curve ONLY
SubjNames=fieldnames(SubjectSNR);
OrderedSubjNames=SubjNames;
OrderedSubjectSNR=orderfields(SubjectSNR,OrderedSubjNames);
OrderedSubjectMR=orderfields(SubjectMR,OrderedSubjNames);
%% Manually Reordering Subjects for Brain Stimulation Study 1 ONLY
SubjNames=fieldnames(SubjectSNR);
OrderedSubjNames={SubjNames{1};SubjNames{5};SubjNames{6};SubjNames{7};SubjNames{8};SubjNames{9};SubjNames{10};
                    SubjNames{11};SubjNames{12};SubjNames{2};SubjNames{3};SubjNames{4}};

OrderedSubjectSNR=orderfields(SubjectSNR,OrderedSubjNames);
OrderedSubjectMR=orderfields(SubjectMR,OrderedSubjNames);
%% Manually Reordering Subjects for Brain Stimulation Study 2 ONLY
SubjNames=fieldnames(SubjectSNR);
OrderedSubjNames={SubjNames{18};SubjNames{6};SubjNames{12};SubjNames{13};SubjNames{7};SubjNames{14};SubjNames{21};
                     SubjNames{15};SubjNames{1};SubjNames{16};SubjNames{8};SubjNames{9};SubjNames{17};SubjNames{2};SubjNames{10};
                     SubjNames{19};SubjNames{11};SubjNames{20};SubjNames{3};SubjNames{4};SubjNames{5}};

OrderedSubjectSNR=orderfields(SubjectSNR,OrderedSubjNames);
OrderedSubjectMR=orderfields(SubjectMR,OrderedSubjNames);
%% Channel-Wise Figure Plotting
channelidx=size(SNRall,2)/2;
SNR3(:,:,1)=SNRall(:,1:channelidx);
SNR3(:,:,2)=SNRall(:,channelidx+1:2*channelidx);
%SNR3(:,:,3)=SNRall(:,2*channelidx+1:3*channelidx);
SNRtotal=mean(SNR3,3);
SNRmean=mean(SNRtotal);
SNRstd=std(SNRtotal);

MR3(:,:,1)=MRall(:,1:channelidx);
MR3(:,:,2)=MRall(:,channelidx+1:2*channelidx);
%MR3(:,:,3)=MRall(:,2*channelidx+1:3*channelidx);
MRtotal=mean(MR3,3);
MRmean=mean(MRtotal);
MRstd=std(MRtotal);

figure
hold on
x=1:size(SNRmean,2);
bar(x,SNRmean,"white")
errorbar(x,SNRmean,SNRstd,"LineStyle","none","color",'black')
xlabel('Channel')
ylabel('Mean SNR')

figure
hold on
bar(x,MRmean,"white")
errorbar(x,MRmean,MRstd,"LineStyle","none","color",'black')
xlabel('Channel')
ylabel('Mean Motion Ratio')

%% Subject-Wise Figure Plotting (Bar Graphs)
MRstd=[];
SNRstd=[];
SNRClass=struct2cell(SubjectSNR);
MRClass=struct2cell(SubjectMR);
SubjNames=fieldnames(SubjectSNR);

for sidx=1:length(SubjNames)
    SNRplot(sidx)=mean(SNRClass{sidx},"all");
    SNRstd(sidx)=std(SNRClass{sidx},0,"all");
    MRplot(sidx)=mean(MRClass{sidx},"all");
    MRstd(sidx)=std(MRClass{sidx},0,"all");
end
x=categorical(SubjNames);
figure
hold on
bar(x,SNRplot,"white")
errorbar(x,SNRplot,SNRstd,"LineStyle","none","color",'black')
xlabel('Subject Label')
ylabel('SNR')

figure
hold on
bar(x,MRplot,"white")
errorbar(x,MRplot,MRstd,"LineStyle","none","color",'black')
xlabel('Subject Label')
ylabel('Motion Ratio')

%% Subject-Wise Figure Plotting (BoxPlots)
SNRClass=struct2cell(OrderedSubjectSNR);
MRClass=struct2cell(OrderedSubjectMR);
SubjNames=OrderedSubjNames;fieldnames(SubjectSNR);
SubjNamesVector=strings(0);
SNRVector=[];
MRVector=[];
for idx1=1:length(SubjNames)
    Subjelem=numel(SNRClass{idx1});
    SNRSubjVector=reshape(SNRClass{idx1},Subjelem,1);
    MRSubjVector=reshape(MRClass{idx1},Subjelem,1);
    SubjNamesVector(end+1:end+Subjelem,:)=repmat(SubjNames{idx1},Subjelem,1);
    SNRVector=[SNRVector;SNRSubjVector];
    MRVector=[MRVector;MRSubjVector];
end

figure
boxplot(SNRVector,SubjNamesVector)
xlabel('Subject Label')
ylabel('SNR')
figure
boxplot(MRVector,SubjNamesVector)
xlabel('Subject Label')
ylabel('Motion Ratio')
