figure
[x,y,z] = meshgrid(1:1:256,1:1:256,1:1:3);
c = mean_data;
xs = 1:1:256;
ys = xs;
zs = 1:1:3;
h = slice(x,y,z,c,xs,ys,zs);
set(h,'FaceColor','interp',...
    'EdgeColor','none')
camproj perspective
box on
colormap hsv
colorbar