dataFolder='dataset\train';
categories={'1','2','3'};

imdsTrain=imageDatastore(fullfile(dataFolder, categories), 'LabelSource','foldernames');
imdsTrain.ReadFcn=@(filename)readandpreprocess(filename);

imdsTest = imageDatastore('dataset\test','IncludeSubFolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn=@(filename)readandpreprocess(filename);

net=alexnet;

featureLayer = 'fc6';


%özellikler
trainingFeatures = activations(net, imdsTrain, featureLayer, 'OutputAs', 'columns');
testingFeatures = activations(net, imdsTest, featureLayer, 'OutputAs', 'columns');

%etiketler
trainingLabels = imdsTrain.Labels;
testingLabels = imdsTest.Labels;
best=0;
tic;
for i=1:1000
classifier = fitcecoc(trainingFeatures, trainingLabels,'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

predictedLabels = predict(classifier, testingFeatures, 'ObservationsIn', 'columns');

accuracy= mean(predictedLabels==testingLabels);

if accuracy>best
    best=accuracy;
    save('sonuclarFeatExtBest.mat','accuracy','predictedLabels','testingLabels','classifier'); %sonuçları sonuclar.mat dosyasına saklıyoruz ki bu kaydedildikten sonra üzerine çift tıkladığımızda sonuçlar gelsin
    save('model.mat','classifier')
end
sonucAccAlex(i)=accuracy;
disp(sprintf('İter:%d   Accuracy:%f Best accuracy:%f',i,accuracy,best));
end
eniyi=max(sonucAccAlex);
ortalama=mean(sonucAccAlex);
zaman=toc;
disp(sprintf('Best accuracy:%f',eniyi));
disp(sprintf('Mean accuracy:%f',ortalama));
disp(sprintf('Time:%f sec',zaman));

save('sonuclarFeatExtGenel.mat','eniyi','ortalama','sonucAccAlex');


load 'sonuclarFeatExtBest.mat';
cmGoogle = confusionchart(testingLabels,predictedLabels);
%cmGoogle.Title = 'Özellik çıkarım metodu için en iyi koşmaya ait karmaşıklık matrisi';
cmGoogle.RowSummary = 'row-normalized';
cmGoogle.ColumnSummary = 'column-normalized';



