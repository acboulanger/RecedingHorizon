function Visualize(filenameIn,  filenameMovie)
    experiment = load(filenameIn);
    
    y = experiment.yconcat;
    u = experiment.uconcat;
    nstep = size(y,1);
    N = experiment.args.N;
    
    yobs = repmat(experiment.args.yobs(1,:),experiment.args.nmaxrh+1,1);
    
    l2norm = sqrt(experiment.args.spacestep*sum((y - yobs).*(y-yobs),2));
    
    
    plot(l2norm);
    
    x = experiment.args.x;
    maxy = max(max(y));
    miny = min(min(y));
    maxu = max(max(u));
    minu = min(min(u));
    
    dt = experiment.args.dt;
    i=1;
    
    while(i<nstep )
        i
        figure(1); 
        set(gcf,'Position',[200,200,1000,500])
        subplot(1,2,1);
        hold on
        plot(x,y(i,:),'LineWidth',2);
        xpatch = [x(floor((N)/12)), x(floor(4*(N)/12)), x(floor(4*(N)/12)), x(floor((N)/12))];
        ypatch = [miny-1,miny-1,maxy+1,maxy+1];
        p = patch(xpatch,ypatch,[0.5,0.5,0.5]);
        p.FaceAlpha = 0.3;
        p.EdgeColor = [0.5,0.5,0.5];
        xpatch2 = [x(floor(8*(N)/12)), x(floor(11*(N)/12)), x(floor(11*(N)/12)), x(floor(8*(N)/12))];
        ypatch2 = [miny-1,miny-1,maxy+1,maxy+1];
        p2 = patch(xpatch2,ypatch2,[0.5,0.5,0.5]);
        p2.FaceAlpha = 0.3;
        p2.EdgeColor = [0.5,0.5,0.5];
        xlabel('x');ylabel('y');
        title('Y');
        axis([x(1),x(end),miny,maxy]);
        hold off

        subplot(1,2,2);
        plot(x,u(i,:),'LineWidth',2);
        xpatch = [x(floor((N)/12)), x(floor(4*(N)/12)), x(floor(4*(N)/12)), x(floor((N)/12))];
        ypatch = [minu-1,minu-1,maxu+1,maxu+1];
        p = patch(xpatch,ypatch,[0.5,0.5,0.5]);
        p.FaceAlpha = 0.3;
        p.EdgeColor = [0.5,0.5,0.5];
        xpatch2 = [x(floor(8*(N)/12)), x(floor(11*(N)/12)), x(floor(11*(N)/12)), x(floor(8*(N)/12))];
        ypatch2 = [minu-1,minu-1,maxu+1,maxu+1];
        p2 = patch(xpatch2,ypatch2,[0.5,0.5,0.5]);
        p2.FaceAlpha = 0.3;
        p2.EdgeColor = [0.5,0.5,0.5];
        xlabel('x');ylabel('u');
        title('U');
        axis([x(1),x(end),minu,maxu]);
        
        t = dt*(i-1);
        str = sprintf('t =%0.2f, ||y - y_{obs}||_{L^2(\\Omega)} = %d',t,l2norm(i));
        suptitle(str);
        
        drawnow()
    
        frame = getframe(1);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if i == 1;
        imwrite(imind,cm,filenameMovie,'gif','DelayTime',0.1,'Loopcount',inf);
        else
        imwrite(imind,cm,filenameMovie,'gif','WriteMode','append','DelayTime',0.1);
        end
    
        if i < 500
            i = i+2;
        elseif i < 5000 
            i = i+20;
        else
            i = i+100;
        end
    end

end