function [uconcat,yconcat,pconcat,args] = RecedingHorizonPeriodic()

    ClearClose();
    args = CreateParameters();
    args.dealiasing = 1;
    
    % observation domain
    args.matrices.Obs = ComputeObservationMatrix(1,args.N,args);

    % control domain
    controldomain = zeros(1,args.N);
    controldomain(1:end) = 1.0;
    %controldomain(floor(5*(args.N+1)/8.0):floor(7*(args.N+1)/8.0)) = 1.0;
    [chi, chiT] = ComputeControlMatrix2(controldomain,args);
    args.matrices.B = chi;
    args.matrices.BT = chiT;
    
    
    %% Uncomment if you want to check gradient/hessian
    u =zeros(args.nmax+1, args.N);
    %u(:,args.N/4+2:end-args.N/4) = 1.0 ;
    for i=1:args.nmax+1
        u(i,:) = exp(-(args.x+5*pi).^2);
    end
    %u = 0.1*ones(args.nmax+1, args.N);
    CheckGradient(u, u, @solveState, @solveAdjoint, ...
    @compute_j, @compute_derivatives_j, args);

    args.kappa = 0.50;
    args.x0 = -2.0;
    args.y0 = 12*args.kappa^2*sech(args.kappa*(args.x - args.x0)).^2;%valeurs aux chebypoints
    args.y0 = args.y0';
    u = zeros(args.nmax+1,args.N);%initialization of the control

    y = solveState(u,args);% one forward simulation for y
    p = solveAdjoint(u,y,args);% one forward simulation for y

    %% Visu
    plottedsteps=1:2:size(y.spatial,1);
    [tg,xg] = meshgrid(args.tdata(plottedsteps),args.x(1:end));
    figure(1);
    surf(xg,tg,y.spatial(plottedsteps,:)');
    xlabel('x');ylabel('Time');zlabel('State variable y');
    title('State Variable y');
    view(-16,10);
    shading interp;
    
    
    plottedsteps=1:2:size(p.spatial,1);
    [tg,xg] = meshgrid(args.tdata(plottedsteps),args.x(1:end));
    figure(2);
    surf(xg,tg,p.spatial(plottedsteps,:)');
    xlabel('x');ylabel('Time');zlabel('Adjoint variable y');
    title('Adjoint Variable y');
    view(-16,10);
    shading interp;

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
    args.N = 160; %number of points
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

    %time argseters
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
    
    % default init
    args.y0 = zeros(1,args.N)';
    args.dy0 = zeros(1,args.N)';
    args.yobs = zeros(args.nmax+1,args.N);
    args.yspecobs = fft(args.yobs,[],2);  

    % Optimization parameters
    args.gamma = 1.0;
    args.epsilon = 1e-12;
    % For fsolve
    args.optimOptState.TolFun = 1e-8;
    args.optimOptState.Jacobian = 'on';
    args.optimOptState.Display = 'off';
    args.optimOptState.Algorithm = 'trust-region-reflective';
    args.optimOptState.JacobMult = @(Jinfo,y,flag)jmfunState(Jinfo,y,flag,args);
    
    args.optimOptAdjoint.TolFun = 1e-8;
    args.optimOptAdjoint.Jacobian = 'on';
    args.optimOptAdjoint.Display = 'off';
    args.optimOptAdjoint.Algorithm = 'trust-region-reflective';
    args.optimOptAdjoint.JacobMult = @(Jinfo,dp,flag)jmfunAdjoint(Jinfo,dp,flag,args);
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
    exp = args.E.*y + 0.5*args.dt*(args.E.*u - args.coeffNL*0.5*args.g.*args.E.*fft(real(ifft(y)).^2) + up);
    if(args.dealiasing==1)
        exp(floor(args.N/8)+2:end - floor(args.N/8)+1) = 0.0;
    end
end

function [F,Jinfo] = fsolverState(y,b,args)
    F = y  + args.coeffNL*0.5*args.dt*0.5*args.g.*fft(real(ifft(y)).^2) - b;
    Jinfo = y;
    if(args.dealiasing==1)
        F(floor(args.N/8)+2:end - floor(args.N/8)+1) = 0.0;
        Jinfo(floor(args.N/8)+2:end - floor(args.N/8)+1) = 0.0;
    end
end

function W = jmfunState(Jinfo,dy,flag,args)
    if(flag > 0)
        W = dy + args.coeffNL*0.5*args.dt*args.g.*fft(real(ifft(Jinfo)).*real(ifft(dy)));
    elseif (flag < 0)
        W = dy - args.coeffNL*0.5*args.dt*fft(real(ifft(Jinfo)).*real(ifft(args.g.*dy)));
    elseif(flag == 0)
        %fprintf('flag0 in state\n')
        W = dy + args.coeffNL*0.5*args.dt*args.g.*fft(real(ifft(Jinfo)).*real(ifft(dy)))...
             - args.coeffNL*0.5*args.dt*fft(real(ifft(Jinfo)).*real(ifft(args.g.*dy)))...
             + args.coeffNL*0.25*(args.dt)^2*fft(real(ifft(Jinfo)).*...
             ifft(-(args.g).^2.*fft(real(ifft(Jinfo)).*real(ifft(dy)))));
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
    %u = ((args.matrices.B)*(u'))';%effect of indicator function
    fftu = fft(u,[],2);
    
    
    yspeci = yspec0;
    %% Time loop
    for i=2:nmax+1
        b = explicitpartState(yspeci,transpose(fftu(i-1,:)), transpose(fftu(i,:)),args);
        yspeci = fsolve(@(x) fsolverState(x,b,args),yspeci,args.optimOptState);
        yi = real(ifft(yspeci));
        %yi = (ifft(yspeci));
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
    %pi = (ifft(pspeci));
    p.spec(1,:) = pspeci;
    p.spatial(1,:) = pi;
    
    %% Time loop
    for i=2:nmax
        rhs = args.matrices.Obs*(yrev(i,:)'- yobsrev(i,:)');
        fftrhs = fft(rhs);
        b = explicitpartAdjoint(pspeci,transpose(yspecrev(i,:)),fftrhs,args);
        pspeci = fsolve(@(x) fsolverAdjoint(x,transpose(yspecrev(i,:)),b,args),pspeci,args.optimOptAdjoint);
        pi = real(ifft(pspeci));
        %pi = (ifft(pspeci));
        p.spec(i,:) = pspeci;
        p.spatial(i,:) = pi;
    end
    
    %last time step
    rhs = args.matrices.Obs*(yrev(end,:)'- yobsrev(end,:)');
    fftrhs = fft(rhs);
    pspeci = args.Einv.*pspeci + args.coeffNL*0.5*args.dt*fft(real(ifft(args.Einv.*args.g.*pspeci)).*real(ifft(transpose(yspecrev(end,:)))))...
         -0.5*args.dt*fftrhs;
    %pspeci = args.Einv.*pspeci - args.coeffNL*args.dt*args.g.*args.Einv.*fft(real(ifft(pspeci)).*real(ifft(yspecrev(end,:)')))...
    %  -0.5*args.dt*fftrhs;
    pi = real(ifft(pspeci));
    %     %pi = (ifft(pspeci));
    p.spec(end,:) = pspeci;
    p.spatial(end,:) = pi;
    p.spec = p.spec(end:-1:1,:);
    p.spatial = p.spatial(end:-1:1,:);
end

function exp = explicitpartAdjoint(p,y,discr,args)
    exp = args.Einv.*p + args.coeffNL*0.5*args.dt*fft(real(ifft(args.g.*args.Einv.*p)).*real(ifft(y)))...
         - args.dt*discr;
     if(args.dealiasing==1)
        exp(floor(args.N/8)+2:end - floor(args.N/8)+1) = 0.0;
     end
end

function [F,Jinfo] = fsolverAdjoint(p,y,b,args)
    F = p - args.coeffNL*0.5*args.dt*fft(real(ifft(args.g.*p)).*real(ifft(y))) - b;
    Jinfo = y;
    if(args.dealiasing==1)
        F(floor(args.N/8)+2:end - floor(args.N/8)+1) = 0.0;
        Jinfo(floor(args.N/8)+2:end - floor(args.N/8)+1) = 0.0;
     end
end

function W = jmfunAdjoint(Jinfo,dp,flag,args)
    if (flag > 0)
        W = dp - args.coeffNL*0.5*args.dt*fft(real(ifft(Jinfo)).*real(ifft(args.g.*dp)));
    elseif (flag < 0)
        W = dp + args.coeffNL*0.5*args.dt*args.g.*fft(real(ifft(Jinfo)).*real(ifft(dp)));
    elseif (flag == 0)
        W = dp - args.coeffNL*0.5*args.dt*fft(real(ifft(Jinfo)).*real(ifft(args.g.*dp)))...
            + args.coeffNL*0.5*args.dt*args.g.*fft(real(ifft(Jinfo)).*real(ifft(dp)))...
            - args.coeffNL*0.25*args.dt^2*args.g.*fft(real(ifft(Jinfo)).*real(ifft(Jinfo)).*real(ifft(args.g.*dp)));
    end
end

function j = compute_j(u,y,args)
    j = 0;
    discr = args.matrices.Obs*(y.spatial(1,:)'-args.yobs(1,:)');
    ymyd = fft(discr);
    j = j+ 0.5*0.5*args.dt*(ymyd'*ymyd);
    for i=2:args.nmax
        discr = args.matrices.Obs*(y.spatial(i,:)'-args.yobs(i,:)');
        ymyd = fft(discr);
        j = j+ 0.5*args.dt*(ymyd'*ymyd);
    end
    discr = args.matrices.Obs*(y.spatial(end,:)'-args.yobs(end,:)');
    ymyd = fft(discr);
    j = j+ 0.5*0.5*args.dt*(ymyd'*ymyd);
end

function dj = compute_derivatives_j(u,y,p,args)%do not forget time in inner product
    %p: row = time, column = space
    %dj = -dt/2*Bstar*p(1+ Einv);
    %dj = ((args.matrices.BT)*(p.spatial)')';%each column is B*p(t_i)
    %pspatial = ifft(p.spec,[],2);
    %pspatial = p.spatial;
    %dj = ((args.matrices.BT)*(pspatial)')';%each column is B*p(t_i)
    %dj = fft(dj,[],2);
    %(p.spec)*args.Einv;
    dj = zeros(size(p.spec));
    dj(1,:) = (args.matrices.BT)*(ifft(transpose(p.spec(2,:)).*args.Einv));
    for i = 2:args.nmax
        dj(i,:) = (args.matrices.BT)*((ifft(transpose(p.spec(i,:)) + transpose(p.spec(i+1,:)).*args.Einv)));
    end
    dj(end,:) = (args.matrices.BT)*((ifft(transpose(p.spec(end,:)))));
    dj = (fft(dj,[],2));
    dj = -0.5*args.dt*dj(:);%makes a vector
end
    