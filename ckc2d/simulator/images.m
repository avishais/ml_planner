clear all
clc

%%

I1 = imread('startI.png');
I2 = imread('goalI.png');

% I1 = imrotate(I1, 180);
% I2 = imrotate(I2, 180);
% I1 = flipdim(I1, 2);
% I2 = flipdim(I2, 2);

I1 = imcrop(I1, [629 44 1102-629 518-44]);
I2 = imcrop(I2, [629 44 1102-629 518-44]);
% 
% I1 = imresize(I1, .8);
% I2 = imresize(I2, .8);
I1 = imresize(I1, [300 474]);
I2 = imresize(I2, [300 474]);

imshow(I1)
%%

for j = 0:3
    for i = 1:size(I1,2)
        I1(1+j,i,:) = [0 0 0];
        I1(end-j,i,:) = [0 0 0];
        I2(1+j,i,:) = [0 0 0];
        I2(end-j,i,:) = [0 0 0];
    end
    
    for i = 1:size(I1,1)
        I1(i,1+j,:) = [0 0 0];
        I1(i,end-j,:) = [0 0 0];
        I2(i,1+j,:) = [0 0 0];
        I2(i,end-j,:) = [0 0 0];
    end
end

I = [I1 I2];

I = insertText(I,[0 -10],'(a)','FontSize',50,'BoxColor',...
    'w','BoxOpacity',0,'TextColor','black');

imshow(I);

imwrite(I, 'envI.png');

