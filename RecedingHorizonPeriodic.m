function [uconcat,yconcat,pconcat,args] = RecedingHorizonPeriodic()

    ClearClose();
    args = CreateParameters();
    
    % observation domain
    args.matrices.Obs = ComputeObservationMatrix(1,args.N,args);

    % control domain
    controldomain = zeros(1,args.N);
    %controldomain(1:end) = 1.0;
    controldomain(floor(3*(args.N+1)/8.0):floor(5*(args.N+1)/8.0)) = 1.0;
    [chi, chiT] = ComputeControlMatrix2(controldomain,args);
    args.matrices.B = chi;
    args.matrices.BT = chiT;
    
    
    %% Uncomment if you want to check gradient/hessian
    u =zeros(args.nmax+1, args.N);
    u(:,args.N/4+2:end-args.N/4) = 1.0 ;
    %for i=1:args.nmax+1
    %    u(i,:) = exp(-(args.x+5*pi).^2);
    %end
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
    args.coeffNL = 0.0;
    args.dealiasing = 1;
    
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
    args.optimOptState.TolPCG = 1e-6;
    args.optimOptState.JacobMult = @(Jinfo,y,flag)jmfunState(Jinfo,y,flag,args);
    
    args.optimOptAdjoint.TolFun = 1e-8;
    args.optimOptAdjoint.Jacobian = 'on';
    args.optimOptAdjoint.Display = 'off';
    args.optimOptAdjoint.Algorithm = 'trust-region-reflective';
    args.optimOptAdjoint.TolPCG = 1e-6;
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
    aux = args.coeffNL*0.5*args.g.*args.E.*fft(real(ifft(y)).^2);
    if(args.dealiasing==1)
        aux(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
    end
    exp = args.E.*y + 0.5*args.dt*(args.E.*u - aux + up);
    if(args.dealiasing==1)
        exp(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
    end
end

function [F,Jinfo] = fsolverState(y,b,args)
    aux = args.coeffNL*0.5*args.dt*0.5*args.g.*fft(real(ifft(y)).^2);
    if(args.dealiasing==1)
       aux(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
    end
    F = y  + aux - b;
    Jinfo = y;
    %if(args.dealiasing==1)
    %   F(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
    %   Jinfo(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
    %end
end

function W = jmfunState(Jinfo,Dy,flag,args)

    for j=1:size(Dy,2)
        dy = Dy(:,j);
        if(flag > 0)
            aux = args.coeffNL*0.5*args.dt*args.g.*fft(real(ifft(Jinfo)).*real(ifft(dy)));
            if(args.dealiasing==1)
                aux(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
            end
            W(:,j) = dy + aux;
        elseif (flag < 0)
            aux = args.coeffNL*0.5*args.dt*fft(real(ifft(Jinfo)).*real(ifft(args.g.*dy)));
            if(args.dealiasing==1)
                aux(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
            end
            W(:,j) = dy - aux;
        elseif(flag == 0)
            %size(args.g)
            %size(Jinfo)
            %size(dy)
            %xxxxxxxxxx = 0
            aux1 = args.coeffNL*0.5*args.dt*args.g.*fft(real(ifft(Jinfo)).*real(ifft(dy)));
            aux2 = args.coeffNL*0.5*args.dt*fft(real(ifft(Jinfo)).*real(ifft(args.g.*dy)));
            aux3 = args.coeffNL*0.25*(args.dt)^2*fft(real(ifft(Jinfo)).*...
                 ifft(-(args.g).^2.*fft(real(ifft(Jinfo)).*real(ifft(dy)))));
            if(args.dealiasing==1)
                aux1(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
                aux2(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
                aux3(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
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
    %u = ((args.matrices.B)*(u'))';%effect of indicator function
    fftu = fft(u,[],2);
    
    
    yspeci = yspec0;
    %% Time loop
    for i=2:nmax+1
        b = explicitpartState(yspeci,transpose(fftu(i-1,:)), transpose(fftu(i,:)),args);
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
        aux(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
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
        aux(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
    end
    exp = args.Einv.*p + aux - args.dt*discr;
    %if(args.dealiasing==1)
    %    exp(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
    %end
end

function [F,Jinfo] = fsolverAdjoint(p,y,b,args)
    aux = args.coeffNL*0.5*args.dt*fft(real(ifft(args.g.*p)).*real(ifft(y)));
    if(args.dealiasing==1)
        aux(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
    end
    F = p - aux - b;
    Jinfo = y;
    %if(args.dealiasing==1)
    %    F(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
    %    Jinfo(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
    %end
end

function W = jmfunAdjoint(Jinfo,Dp,flag,args)

for j=1:size(Dp,2)
    dp = Dp(:,j);
    if (flag > 0)
        aux = args.coeffNL*0.5*args.dt*fft(real(ifft(Jinfo)).*real(ifft(args.g.*dp)));
        if(args.dealiasing==1)
            aux(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
        end
        W(:,j) = dp - aux;
    elseif (flag < 0)
        aux = args.coeffNL*0.5*args.dt*args.g.*fft(real(ifft(Jinfo)).*real(ifft(dp)));
        if(args.dealiasing==1)
            aux(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
        end
        W(:,j) = dp + aux;
    elseif (flag == 0)
        aux1 = args.coeffNL*0.5*args.dt*fft(real(ifft(Jinfo)).*real(ifft(args.g.*dp)));
        aux2 = args.coeffNL*0.5*args.dt*args.g.*fft(real(ifft(Jinfo)).*real(ifft(dp)));
        aux3 = args.coeffNL*0.25*args.dt^2*args.g.*fft(real(ifft(Jinfo)).*real(ifft(Jinfo)).*real(ifft(args.g.*dp)));
       if(args.dealiasing==1)
            aux1(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
            aux2(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
            aux3(floor(args.N/4)+2:end - floor(args.N/4)+1) = 0.0;
       end
        W(:,j) = dp - aux1...
        + aux2...
        - aux3;
    end
end    
end

function j = compute_j(u,y,args)
    for i=1:args.nmax+1
    u(i,:) = args.matrices.B*u(i,:)';
    end
    fftu = fft(u,[],2);

    j = 0;
    discr = args.matrices.Obs*(y.spatial(1,:)'-args.yobs(1,:)');
    ymyd = fft(discr);
    j = j+ 0.5*0.5*args.dt*(ymyd'*ymyd);
    %j = j+ 0.5*args.gamma*0.5*args.dt*fftu(1,:)*fftu(1,:)';
    for i=2:args.nmax
        discr = args.matrices.Obs*(y.spatial(i,:)'-args.yobs(i,:)');
        ymyd = fft(discr);
        j = j+ 0.5*args.dt*(ymyd'*ymyd);
        %j = j+ 0.5*args.gamma*args.dt*fftu(i,:)*fftu(i,:)';
    end
    discr = args.matrices.Obs*(y.spatial(end,:)'-args.yobs(end,:)');
    ymyd = fft(discr);
    j = j+ 0.5*0.5*args.dt*(ymyd'*ymyd);
    %j = j+ 0.5*args.gamma*0.5*args.dt*fftu(end,:)*fftu(end,:)';
end

function dj = compute_derivatives_j(u,y,p,args)%do not forget time in inner product
    %p: row = time, column = space
    %dj = -dt/2*Bstar*p(1+ Einv);
    
    for i=1:args.nmax+1
    u(i,:) = args.matrices.B*u(i,:)';
    end
    fftu = fft(u,[],2);

    j = 0;
    
    dj = zeros(size(p.spec));
    dj(1,:) = -0.5*args.dt*(args.matrices.BT)*(ifft(transpose(p.spec(2,:)).*args.Einv));
    for i = 2:args.nmax
        dj(i,:) = -0.5*args.dt*(args.matrices.BT)*((ifft(transpose(p.spec(i,:)) + transpose(p.spec(i+1,:)).*args.Einv)));
    end
    dj(end,:) = -0.5*args.dt*(args.matrices.BT)*((ifft(transpose(p.spec(end,:)))));
    dj = (fft(dj,[],2));
    
    %%second part
%     dj(1,:) = dj(1,:) + 0.5*args.dt*args.gamma*fftu(1,:);
%     for i = 2:args.nmax
%         dj(i,:) = dj(i,:) + args.dt*args.gamma*fftu(i,:);
%     end
%     dj(end,:) = dj(end,:)  + 0.5*args.dt*args.gamma*fftu(end,:);
     dj = dj(:);
end
    