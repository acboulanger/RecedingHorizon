function [uconcat,yconcat,pconcat,args] = RecedingHorizonPeriodic()

    ClearClose();
    args = CreateParameters();
    
    % observation domain
    args.matrices.Obs = ComputeObservationMatrix(1,args.N,args);

    % control domain
    controldomain = zeros(1,args.N);
    controldomain(1:end) = 1.0;
    %controldomain(floor(3*(args.N)/8.0):floor(5*(args.N)/8.0)) = 1.0;
    [chi, chiT] = ComputeControlMatrix2(controldomain,args);
    args.matrices.B = chi;
    args.matrices.BT = chiT;
    
    
    %% Uncomment if you want to check gradient/hessian
%     u =zeros(args.nmax+1, args.N);
%     u(:,args.N/4+2:end-args.N/4) = 0.1 ;
%     %for i=1:args.nmax+1
%     %    u(i,:) = exp(-(args.x+5*pi).^2);
%     %end
%     %u = 0.1*ones(args.nmax+1, args.N);
%      CheckGradient(u, u, @solveState, @solveAdjoint, ...
%      @compute_j, @compute_derivatives_j, args);

    args.kappa = 0.50;
    args.x0 = -2.0;
    args.y0 = 12*args.kappa^2*sech(args.kappa*(args.x - args.x0)).^2;%valeurs aux chebypoints
    args.y0 = args.y0';
    %args.yobs = args.D/args.N*fft(args.y0)'*fft(args.y0)*ones(size(args.y0));
    args.yobs = zeros(args.nmax+1, args.N);
%     u = zeros(args.nmax+1,args.N);%initialization of the control
% 
%     y = solveState(u,args);% one forward simulation for y
%     p = solveAdjoint(u,y,args);% one forward simulation for y

    %% Visu
%     plottedsteps=1:2:size(y.spatial,1);
%     [tg,xg] = meshgrid(args.tdata(plottedsteps),args.x(1:end));
%     figure(1);
%     surf(xg,tg,y.spatial(plottedsteps,:)');
%     xlabel('x');ylabel('Time');zlabel('State variable y');
%     title('State Variable y');
%     view(-16,10);
%     shading interp;
%     
%     
%     plottedsteps=1:2:size(p.spatial,1);
%     [tg,xg] = meshgrid(args.tdata(plottedsteps),args.x(1:end));
%     figure(2);
%     surf(xg,tg,p.spatial(plottedsteps,:)');
%     xlabel('x');ylabel('Time');zlabel('Adjoint variable y');
%     title('Adjoint Variable y');
%     view(-16,10);
%     shading interp;
    
    
    %% Receding Horizon
    yconcat= zeros(floor(args.Tinf/args.dt)+1, args.N);
    pconcat= zeros(floor(args.Tinf/args.dt)+1, args.N);
    uconcat= zeros(floor(args.Tinf/args.dt)+1, args.N);
    for irh=1:args.nrecinf
        if(irh>1)%initial condition is previous result at time = delta
            args.y0 = ykeep(end,:);
            args.y0 = args.y0';
        end
        %run optimization process bis T;
        [y,p,u,args] = solveOptimization((irh-1)*args.deltarh,args);
        
        %keep data only until deltarh
        ykeep = y.spatial(1:args.nkeep,:);
        pkeep = p.spatial(1:args.nkeep,:);
        ukeep = u(1:args.nkeep,:);
        
        %concatenate results
        if(irh==1)
          yconcat(1:(args.nkeep-1),:) =  ykeep(1:end-1,:);
          pconcat(1:(args.nkeep-1),:) =  pkeep(1:end-1,:); 
          uconcat(1:(args.nkeep-1),:) = ukeep(1:end-1,:);
        else
            yconcat((args.nkeep-1)*(irh-1) + (1:(args.nkeep-1)),:) = ykeep(1:end-1,:);%last step counts as initial step in next
            pconcat((args.nkeep-1)*(irh-1) + (1:(args.nkeep-1)),:) = pkeep(1:end-1,:);
            uconcat((args.nkeep-1)*(irh-1) + (1:(args.nkeep-1)),:) = ukeep(1:end-1,:);
        end
        if(irh==(args.nrecinf))
            yconcat(end,:) = ykeep(end,:);
            pconcat(end,:) = pkeep(end,:);
            uconcat(end,:) = ukeep(end,:);
        end
        myvisu(1,yconcat,pconcat,uconcat,args);
        %visunormL2(2,yconcat,args);
    end%end loop on deltas
    %myvisu(1,yconcat,pconcat,uconcat,args);
    %plot evolution of L2 norms
    %visunormL2(2,yconcat,args);
    %visunormL2(3,uconcat,args);   

end

function ClearClose()   
    % Close all figures including those with hidden handles
    close all hidden;

    % Store all the currently set breakpoints in a variable
    temporaryBreakpointData=dbstatus('-completenames');

    % Clear functions and their persistent variables (also clears breakpoints 
    % set in functions)
    clear functions;

    % Restore the previously set breakpoints
    dbstop(temporaryBreakpointData);

    % Clear global variables
    clear global;

    % Clear variables (including the temporary one used to store breakpoints)
    clear variables;
end

function args = CreateParameters()
    % Mesh
    args.D = 5*pi; %domain is -5pi..5pi
    args.N = 256; %number of points
    args.x = linspace(-args.D,args.D,args.N);
    args.k = [0:args.N/2-1 0 -args.N/2+1:-1]*2*pi/(2*args.D); % frequency grid
    args.npoints = size(args.x,2);
    args.spacestep = args.x(2)-args.x(1);
    args.ncells = args.npoints-1;
    
    %Receding horizon
    args.deltarh = 1.0;
    args.T = 1.5;
    args.Tinf = 500.0;
    args.nrecinf = floor(args.Tinf/args.deltarh);

    %time
    args.dt = 0.01;% time step for simulation
    args.tmax = args.T;% maximum time for simulation
    args.nmax = round(args.tmax/args.dt);% induced number of time steps
    args.tdata = args.dt*(0:1:(args.nmax+1));
    args.maxiter = 1e3;
    args.nkeep = floor(args.deltarh/args.dt)+1;
    args.nmaxrh = round(args.Tinf/args.dt);% induced number of time steps
    args.tdatarh = args.dt*(0:1:(args.nmaxrh+1));
    
    %% Useful constants
    ik3 = 1i*(args.k' - (args.k').^3);
    args.g = 1i*(args.k');
    args.E = exp(-args.dt*ik3);
    args.Einv = exp(args.dt*ik3);

    % Misc
    args.coeffNL = 1.0;
    args.dealiasing = 0;
    args.arraydealiasing = floor(args.N/4):floor(3*args.N/4);
    
    % default init
    args.y0 = zeros(1,args.N)';
    args.dy0 = zeros(1,args.N)';
    args.yobs = zeros(args.nmax+1,args.N);
    args.yspecobs = fft(args.yobs,[],2);  

    % Optimization parameters
    args.gamma = 1e-3;
    args.tol = 1e-2;
    args.beta = 0.5;
    args.zeta = 0.1;
    
    % For fsolve   
    args.optimOptState.TolFun = 1e-8;
    args.optimOptState.Jacobian = 'on';
    args.optimOptState.Display = 'off';
    args.optimOptState.Algorithm = 'trust-region-reflective';
    args.optimOptState.TolPCG = 1e-6;
    args.optimOptState.JacobMult = @(Jinfo,y,flag)jmfunState(Jinfo,y,flag,args);
    
    args.optimOptAdjoint.TolFun = 1e-8;
    args.optimOptAdjoint.Jacobian = 'on';
    args.optimOptAdjoint.Display = 'off';
    args.optimOptAdjoint.Algorithm = 'trust-region-reflective';
    args.optimOptAdjoint.TolPCG = 1e-6;
    args.optimOptAdjoint.JacobMult = @(Jinfo,dp,flag)jmfunAdjoint(Jinfo,dp,flag,args);
    
    %matrices
    args.MassS = 2*args.D/(args.N)^2*speye(args.N);
    args.MassT = args.dt*speye(args.nmax+1);
    args.MassT(1,1) = 0.5*args.dt;
    args.MassT(end,end) = 0.5*args.dt;
    % construct the matrix A for the inner product in H (space and time)
    % (u,v)_H = u'Av
    args.Mass = kron(args.MassS,args.MassT);%should be diag(MassT(i)*MassS)
end

function [Obs] = ComputeObservationMatrix(i1,i2,args)
    observationdomain = i1:i2;
    Obs = zeros(args.N);
    for i=1:size(observationdomain,2)
        Obs(observationdomain(i), observationdomain(i)) = 1;
    end
end

function [B,BT] = ComputeControlMatrix(i1,i2,args)
    controldomain = i1:i2;
    B = zeros(args.N+1);
    for i=1:size(controldomain,2)
        B(controldomain(i), controldomain(i)) = 1.0;
    end
BT = B';  
end

function [B,BT] = ComputeControlMatrix2(controldomain,args)
    B = zeros(args.N);
    for i=1:(args.N)
        if(controldomain(i)==1)
            B(i, i) = 1.0;
        end
    end
BT = B';  
end

function exp = explicitpartState(y,u,up,args)
    aux = args.coeffNL*0.5*args.g.*args.E.*fft(real(ifft(y)).^2);
    if(args.dealiasing==1)
        aux(args.arraydealiasing) = 0.0;
    end
    exp = args.E.*y + 0.5*args.dt*(args.E.*u - aux + up);
end

function [F,Jinfo] = fsolverState(y,b,args)
    aux = args.coeffNL*0.5*args.dt*0.5*args.g.*fft(real(ifft(y)).^2);
    if(args.dealiasing==1)
       aux(args.arraydealiasing) = 0.0;
    end
    F = y  + aux - b;
    Jinfo = y;
end

function W = jmfunState(Jinfo,Dy,flag,args)

    for j=1:size(Dy,2)
        dy = Dy(:,j);
        if(flag > 0)
            aux = args.coeffNL*0.5*args.dt*args.g.*fft(real(ifft(Jinfo)).*real(ifft(dy)));
            if(args.dealiasing==1)
                aux(args.arraydealiasing) = 0.0;
            end
            W(:,j) = dy + aux;
        elseif (flag < 0)
            aux = args.coeffNL*0.5*args.dt*fft(real(ifft(Jinfo)).*real(ifft(args.g.*dy)));
            if(args.dealiasing==1)
                aux(args.arraydealiasing) = 0.0;
            end
            W(:,j) = dy - aux;
        elseif(flag == 0)
            aux1 = args.coeffNL*0.5*args.dt*args.g.*fft(real(ifft(Jinfo)).*real(ifft(dy)));
            aux2 = args.coeffNL*0.5*args.dt*fft(real(ifft(Jinfo)).*real(ifft(args.g.*dy)));
            aux3 = args.coeffNL*0.25*(args.dt)^2*fft(real(ifft(Jinfo)).*...
                 ifft(-(args.g).^2.*fft(real(ifft(Jinfo)).*real(ifft(dy)))));
            if(args.dealiasing==1)
                aux1(args.arraydealiasing) = 0.0;
                aux2(args.arraydealiasing) = 0.0;
                aux3(args.arraydealiasing) = 0.0;
            end
            W(:,j) = dy + aux1...
            - aux2...
            + aux3;
        end
    end
end

function [y] = solveState(u,args)%CN scheme in time
    % parameters
    dt=args.dt;
    nmax=args.nmax;
    N=args.N;
    
    % state variables
    y.spatial = zeros(nmax+1,N);% spatial domain
    y.spec = zeros(nmax+1,N);% spectral domain
    
    % initialization
    yspec0 = fft(args.y0);
    y.spatial(1,:) = args.y0;
    y.spec(1,:) = yspec0;
    for i=1:args.nmax+1
        u(i,:) = args.matrices.B*u(i,:)';
    end
    fftu = fft(u,[],2);
    
    
    yspeci = yspec0;
    %% Time loop
    for i=2:nmax+1
        b = explicitpartState(yspeci,transpose(fftu(i-1,:)),transpose(fftu(i,:)),args);
        [yspeci,fmin,exitflag] = fsolve(@(x) fsolverState(x,b,args),yspeci,args.optimOptState);
        yi = real(ifft(yspeci));
        y.spec(i,:) = yspeci;
        y.spatial(i,:) = yi;
    end
end

function p = solveAdjoint(u,y,args)%CN scheme in time
    % parameters
    dt=args.dt;
    nmax=args.nmax;
    N=args.N;
    coeffNL = args.coeffNL;
    
    % state variables
    p.spatial = zeros(nmax+1,N);
    p.spec = zeros(nmax+1,N);

    yrev = y.spatial(end:-1:1,:);
    yobsrev = args.yobs(end:-1:1,:);
    yspecrev = y.spec(end:-1:1,:);

    %initial condition(implicit given only)
    rhs = args.matrices.Obs*(y.spatial(end,:)'- args.yobs(end,:)');
    fftrhs = fft(rhs);
    b = -0.5*args.dt*fftrhs;
    pspeci = fsolve(@(x) fsolverAdjoint(x,transpose(y.spec(end,:)),b,args),transpose(p.spec(1,:)),args.optimOptAdjoint);
    pi = real(ifft(pspeci));
    p.spec(1,:) = pspeci;
    p.spatial(1,:) = pi;
    
    %% Time loop
    for i=2:nmax
        rhs = args.matrices.Obs*(yrev(i,:)'- yobsrev(i,:)');
        fftrhs = fft(rhs);
        b = explicitpartAdjoint(pspeci,transpose(yspecrev(i,:)),fftrhs,args);
        pspeci = fsolve(@(x) fsolverAdjoint(x,transpose(yspecrev(i,:)),b,args),pspeci,args.optimOptAdjoint);
        pi = real(ifft(pspeci));
        p.spec(i,:) = pspeci;
        p.spatial(i,:) = pi;
    end
    
    %last time step
    rhs = args.matrices.Obs*(yrev(end,:)'- yobsrev(end,:)');
    fftrhs = fft(rhs);
    aux = args.coeffNL*0.5*args.dt*fft(real(ifft(args.Einv.*args.g.*pspeci)).*real(ifft(transpose(yspecrev(end,:)))));
    if(args.dealiasing==1)
        aux(args.arraydealiasing) = 0.0;
    end
    pspeci = args.Einv.*pspeci + aux - 0.5*args.dt*fftrhs;
    pi = real(ifft(pspeci));
    p.spec(end,:) = pspeci;
    p.spatial(end,:) = pi;
    p.spec = p.spec(end:-1:1,:);
    p.spatial = p.spatial(end:-1:1,:);
end

function exp = explicitpartAdjoint(p,y,discr,args)
    aux = args.coeffNL*0.5*args.dt*fft(real(ifft(args.g.*args.Einv.*p)).*real(ifft(y)));
    if(args.dealiasing==1)
        aux(args.arraydealiasing) = 0.0;
    end
    exp = args.Einv.*p + aux - args.dt*discr;
    %if(args.dealiasing==1)
    %    exp(args.arraydealiasing) = 0.0;
    %end
end

function [F,Jinfo] = fsolverAdjoint(p,y,b,args)
    aux = args.coeffNL*0.5*args.dt*fft(real(ifft(args.g.*p)).*real(ifft(y)));
    if(args.dealiasing==1)
        aux(args.arraydealiasing) = 0.0;
    end
    F = p - aux - b;
    Jinfo = y;
    %if(args.dealiasing==1)
    %    F(args.arraydealiasing) = 0.0;
    %    Jinfo(args.arraydealiasing) = 0.0;
    %end
end

function W = jmfunAdjoint(Jinfo,Dp,flag,args)

for j=1:size(Dp,2)
    dp = Dp(:,j);
    if (flag > 0)
        aux = args.coeffNL*0.5*args.dt*fft(real(ifft(Jinfo)).*real(ifft(args.g.*dp)));
        if(args.dealiasing==1)
            aux(args.arraydealiasing) = 0.0;
        end
        W(:,j) = dp - aux;
    elseif (flag < 0)
        aux = args.coeffNL*0.5*args.dt*args.g.*fft(real(ifft(Jinfo)).*real(ifft(dp)));
        if(args.dealiasing==1)
            aux(args.arraydealiasing) = 0.0;
        end
        W(:,j) = dp + aux;
    elseif (flag == 0)
        aux1 = args.coeffNL*0.5*args.dt*fft(real(ifft(Jinfo)).*real(ifft(args.g.*dp)));
        aux2 = args.coeffNL*0.5*args.dt*args.g.*fft(real(ifft(Jinfo)).*real(ifft(dp)));
        aux3 = args.coeffNL*0.25*args.dt^2*args.g.*fft(real(ifft(Jinfo)).*real(ifft(Jinfo)).*real(ifft(args.g.*dp)));
       if(args.dealiasing==1)
            aux1(args.arraydealiasing) = 0.0;
            aux2(args.arraydealiasing) = 0.0;
            aux3(args.arraydealiasing) = 0.0;
       end
        W(:,j) = dp - aux1...
        + aux2...
        - aux3;
    end
end    
end

function j = compute_j(u,y,args)
%     for i=1:args.nmax+1
%     u(i,:) = args.matrices.B*u(i,:)';
%     end
%     fftu = fft(u,[],2);
% 
%     j = 0;
%     discr = args.matrices.Obs*(y.spatial(1,:)'-args.yobs(1,:)');
%     ymyd = fft(discr);
%     j = j+ 0.5*0.5*args.dt*(2*args.D/args.N^2)*(ymyd'*ymyd);
%     j = j+ 0.5*args.gamma*0.5*args.dt*(2*args.D/args.N^2)*(fftu(1,:)*fftu(1,:)');
%     for i=2:args.nmax
%         discr = args.matrices.Obs*(y.spatial(i,:)'-args.yobs(i,:)');
%         ymyd = fft(discr);
%         j = j+ 0.5*args.dt*(2*args.D/args.N^2)*(ymyd'*ymyd);
%         j = j+ 0.5*args.gamma*args.dt*(2*args.D/args.N^2)*(fftu(i,:)*fftu(i,:)');
%     end
%     discr = args.matrices.Obs*(y.spatial(end,:)'-args.yobs(end,:)');
%     ymyd = fft(discr);
%     j = j+ 0.5*0.5*args.dt*(2*args.D/args.N^2)*(ymyd'*ymyd);
%     j = j+ 0.5*args.gamma*0.5*args.dt*(2*args.D/args.N^2)*(fftu(end,:)*fftu(end,:)');
    
    %tracking term
    track = zeros(args.nmax+1,args.N);
    for i=1:args.nmax+1
        track(i,:) = fft(args.matrices.Obs*(y.spatial(i,:)'-args.yobs(i,:)'));
    end
    track = track(:);
    
    %reg term
    for i=1:args.nmax+1
    u(i,:) = args.matrices.B*u(i,:)';
    end
    fftu = fft(u,[],2);
    fftu = fftu(:);
    
    M = args.Mass;
    
    j = real(0.5*(track'*M*track) + 0.5*args.gamma*(fftu'*M*fftu));
end

function [djcol,djmat] = compute_derivatives_j(u,y,p,args)%do not forget time in inner product
    %p: row = time, column = space
    %dj = -dt/2*Bstar*p(1+ Einv);
    
%     for i=1:args.nmax+1
%     u(i,:) = args.matrices.B*u(i,:)';
%     end
%     fftu = fft(u,[],2);
%     fftu = fftu(:);
%     
%     dj = zeros(size(p.spec));
%     dj(1,:) = -0.5*args.dt*(args.matrices.BT)*(ifft(transpose(p.spec(2,:)).*args.Einv));
%     for i = 2:args.nmax
%         dj(i,:) = -0.5*args.dt*(args.matrices.BT)*((ifft(transpose(p.spec(i,:)) + transpose(p.spec(i+1,:)).*args.Einv)));
%     end
%     dj(end,:) = -0.5*args.dt*(args.matrices.BT)*((ifft(transpose(p.spec(end,:)))));
%     dj = (fft(dj,[],2));
%     
%     %%second part
%     dj(1,:) = dj(1,:) + 0.5*args.dt*args.gamma*(2*args.D/args.N^2)*fftu(1,:);
%     for i = 2:args.nmax
%         dj(i,:) = dj(i,:) + args.dt*args.gamma*(2*args.D/args.N^2)*fftu(i,:);
%     end
%     dj(end,:) = dj(end,:)  + 0.5*args.dt*args.gamma*(2*args.D/args.N^2)*fftu(end,:);
%     dj2 = dj;
%     dj = dj(:);
    
    %%reg term
    for i=1:args.nmax+1
        u(i,:) = args.matrices.B*u(i,:)';
    end
    fftu = fft(u,[],2);
    
    %%track term
    dtrack = zeros(args.nmax+1,args.N);
    dtrack(1,:) = -(args.matrices.BT)*(ifft(transpose(p.spec(2,:)).*args.Einv));
    for i = 2:args.nmax
         dtrack(i,:) = -0.5*(args.matrices.BT)*((ifft(transpose(p.spec(i,:)) + transpose(p.spec(i+1,:)).*args.Einv)));
    end
    dtrack(end,:) = -(args.matrices.BT)*((ifft(transpose(p.spec(end,:)))));
    dtrack = (fft(dtrack,[],2));

    djcol = dtrack(:) + args.gamma*fftu(:);%result under column form
    djmat = dtrack + args.gamma*fftu;%result under matrix form row = time, col = space
end

function [y,p,u,args] = solveOptimization(t0,args)

    %params and init
    gamma = args.gamma;
    M = args.Mass;
    abstol=1e-6;
    
    %%  Start of Gradient descent strategy
    fprintf('Gradient descent strategy...\n');
    fprintf('gamma = %d , t_0 = %d, delta = %d\n', ...
        gamma, t0, args.deltarh);
    
    uold = zeros(args.nmax+1, args.N);%initial guess spectral space
    uspaceold = real(ifft(uold,[],2));%initial guess spatial space
    
    yold = solveState(uspaceold,args);% one forward simulation for y
    jold = compute_j(uspaceold,yold,args)
    pold = solveAdjoint(uspaceold,yold,args);% one forward simulation for y

    [grad,grad2] = compute_derivatives_j(uspaceold,yold,pold,args);
    gradold = grad;
    ngrad = sqrt(real(grad'*M*grad))
    
    count = 0;
    while(ngrad > args.tol)
        
        if(count==0)
           stepsize = 1.0/ngrad;
        end
    
        u = uold - stepsize*grad2;%gradient change in spectral space
        uspace = real(ifft(u,[],2));%initial guess spatial space
        y = solveState(uspace,args);% one forward simulation for y
        j = compute_j(uspace,y,args)
        
        %%linesearch
        m = 0;
        while( (j - jold) > -args.zeta*stepsize*ngrad*ngrad && m < 30)
            m = m+1
            stepsize = stepsize*args.beta;
            u = uold - stepsize*grad2;%gradient change in spectral space
            uspace = real(ifft(u,[],2));%go back to spatial space
            y = solveState(uspace,args);%
            j = compute_j(uspace,y,args);
        end
        p = solveAdjoint(uspace,y,args);% one forward simulation for y
        myvisu(1,y.spatial,p.spatial,uspace,args)

        [grad,grad2] = compute_derivatives_j(uspace,y,p,args);
        ngrad = sqrt(real(grad'*M*grad))
        
        %stepsize
        s = u - uold;
        s = s(:);
        g = grad - gradold;%already in column
        stg = real(g'*M*s);
        if(stg > 0)
            disp('stg > 0');
            if mod(count,2)
                sts = real(s'*M*s);
                stepsize = sts/stg;
            else
                gtg = real(g'*M*g);
                stepsize = stg/gtg;
            end
        else
            disp('stg < 0');
            stepsize = 1.0/ngrad;
        end
        
        uold = u;
        jold = j;
        gradold = grad;
        
        count = count+1;
    end
end

function visunormL2(nfig,y,args)
    n = size(y,1);
    L2NormInSpace = zeros(1,n);
    for i=1:n
        discr = y(i,:)';
        ymyd = fft(discr);
        L2NormInSpace(i) = sqrt(ymyd'*args.MassS*ymyd);
    end
    figure(nfig);
    clf(nfig);
    hold on;
    plot(L2NormInSpace);
    xlabel('x');ylabel('||y||_{L^2(I)}');
    hold off;
end

function myvisu(nfig,y,p,q,args)
    %% 3D - Vizualization
    figure(nfig);
    plottedsteps=1:2:size(y,1);
    [tg,xg] = meshgrid(args.tdatarh(plottedsteps),args.x(1:end));
    
    subplot(2,2,1), surf(xg,tg,y(plottedsteps,:)');
    xlabel('x');ylabel('Time');zlabel('y');
    title('State Variable y');
    %axis([-16,16,0,0.5,-1.5,1.5]);
    view(-8,40);
    shading interp;
    
    subplot(2,2,2), surf(xg,tg,p(plottedsteps,:)');
    xlabel('x');ylabel('Time');zlabel('p');
    title('Adjoint state');
    %axis([-16,16,0,0.5,-0.1,0.1]);
    view(-8,40);
    shading interp;
    
    subplot(2,2,3), surf(xg,tg,q(plottedsteps,:)');
    xlabel('x');ylabel('Time');zlabel('u');
    title('Current Control');
    %axis([-16,16,0,0.5,-2,3]);
    view(-8,40);
    shading interp;
    
    drawnow();
end
    