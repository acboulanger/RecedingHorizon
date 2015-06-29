function [uconcat,yconcat,pconcat,args] = RecedingHorizonPeriodic()

    ClearClose();
    args = CreateParameters();

    args.kappa = 0.70;
    args.x0 = -2.0;
    args.y0 = 12*args.kappa^2*sech(args.kappa*(args.x - args.x0)).^2;%valeurs aux chebypoints
    args.y0 = args.y0';
    u = zeros(args.nmax+1,args.N);%initialization of the control

    y = solveState(u,args);% one forward simulation for y
    
    %% Visu
    plottedsteps=1:2:size(y.spatial,1);
    [tg,xg] = meshgrid(args.tdata(plottedsteps),args.x(1:end));
    surf(xg,tg,u(plottedsteps,:)');
    xlabel('x');ylabel('Time');zlabel('u');                   
    title('Source term');
    view(-16,10);
    shading interp;
    figure(1);
    surf(xg,tg,y.spatial(plottedsteps,:)');
    xlabel('x');ylabel('Time');zlabel('State variable y');
    title('State Variable y');
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
    args.g = -0.5*1i*args.k';
    args.E = exp(-args.dt*ik3);
    args.Einv = exp(args.dt*ik3);


    % Optimization parameters
    args.alpha = 0.0;
    args.epsilon = 1e-12;
    % For fsolve
    args.optimOptState.TolFun = 1e-5;
    args.optimOptState.Jacobian = 'on';
    args.optimOptState.Display = 'off';
    args.optimOptState.Algorithm = 'trust-region-reflective';
    args.optimOptState.JacobMult = @(Jinfo,y,flag)jmfunState(Jinfo,y,flag,args);
    
    args.optimOptAdjoint.TolFun = 1e-5;
    args.optimOptAdjoint.Jacobian = 'on';
    args.optimOptAdjoint.Display = 'off';
    args.optimOptAdjoint.Algorithm = 'trust-region-reflective';
    args.optimOptAdjoint.JacobMult = @(Jinfo,y,flag)jmfunadjoint(Jinfo,y,flag,args);

    % Trust region Steihaug globalization
    args.gamma =1.0;
    args.sigma = 10.0;
    args.sigmamax = 100.0;

    % Misc
    args.coeffNL = 1.0;
    
    % default init
    args.y0 = zeros(1,args.N)';
    args.dy0 = zeros(1,args.N)';
    args.yobs = zeros(args.nmax+2,args.N);
    args.yspecobs = zeros(args.nmax+2,args.N)';
    args.q = 0.0*ones(args.nmax+2, args.N);
  
end

function exp = explicitpartState(y,u,up,args)
    exp = args.E.*y + 0.5*args.dt*(args.E.*u + args.coeffNL*args.g.*args.E.*fft(ifft(y).^2) + up);
end

function [F,Jinfo] = fsolverState(y,b,args)
    F = y - args.coeffNL*0.5*args.dt*args.g.*fft(ifft(y).^2) - b;
    Jinfo = y;
end

function W = jmfunState(Jinfo,dy,flag,args)
    if(flag>0)
        W = dy - args.dt*args.g.*fft(ifft(Jinfo).*ifft(dy));
    elseif (flag < 0)
        W = dy + args.dt*args.g.*fft(ifft(Jinfo).*ifft(dy));
    elseif flag == 0 
        W = dy + 0.25*(args.dt)*args.dt*(args.k').^2.*fft(ifft(Jinfo).*ifft(Jinfo).*ifft(dy));
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
    %u = ((args.matrices.B)*(u'))';%effect of indicator function
    fftu = fft(u,[],2);
    
    
    yspeci = yspec0;
    %% Time loop
    for i=2:nmax
        b = explicitpartState(yspeci,fftu(i,:)', fftu(i+1,:)',args);
        yspeci = fsolve(@(x) fsolverState(x,b,args),yspeci,args.optimOptState);
        yi = real(ifft(yspeci));
        y.spec(i+1,:) = yspeci;
        y.spatial(i+1,:) = yi;
    end
end

function p = solveAdjoint(u,y,args)%CN scheme in time
    % parameters
    dt=args.dt;
    nmax=args.nmax;
    N=args.N;
    coeffNL = args.coeffNL;
    matrices = args.matrices;
    
    % state variables
    p.spatial = zeros(nmax+1,N+1);
    p.spec = zeros(nmax+1,N-2);

    yrev = y.spatial(end:-1:1,:);
    yobsrev = args.yobs(end:-1:1,:);
    yspecrev = y.spec(end:-1:1,:);

    
    rhs = args.matrices.Obs*(y.spatial(end,:)'- args.yobs(end,:)');
    fftrhs = fft(rhs);
    
    %initial condition(implicit given only)
    pspeci = yspec0;
    pi = real(ifft(pspeci));
    p.spec(1,:) = pspeci;
    p.spatial(1,:) = pi;
    
    %% Time loop
    for i=2:nmax
        rhs = args.matrices.Obs*(yrev(i,:)'- args.yobsrev(i,:)');
        fftrhs = fft(rhs);
        b = explicitpartadjoint(pspeci,yrev(i,:), fftrhs,args);
        pspeci = fsolve(@(x) fsolverAdjoint(x,b,args),pspeci,args.optimOptAjoint);
        pi = real(ifft(yspeci));
        p.spec(i,:) = pspeci;
        p.spatial(i,:) = pi;
    end
    p.spec = p.spec(end:-1:1,:);
    p.spatial = p.spatial(end:-1:1,:);
end

function exp = explicitpartAdjoint(p,y,disr,args)
    exp = args.Einv.*p - args.coeffNL*args.dt*args.g.*args.Einv.*fft(ifft(p).*ifft(y))...
         - args.dt*discr;
end

function [F,Jinfo] = fsolverAdjoint(p,y,b,args)
    F = p + args.coeffNL*args.dt*args.g.*fft(ifft(p).*ifft(y)) - b;
    Jinfo = y;
end

function W = jmfunadjoint(Jinfo,dp,flag,args)
    if(flag>0)
        W = dp + args.coeffNL*args.dt*args.g.*fft(ifft(Jinfo).*ifft(dp));
    elseif (flag < 0)
        W = dp - args.coeffNL*args.dt*args.g.*fft(ifft(Jinfo).*ifft(dp));
    elseif flag == 0 
        W = dp + 0.25*(args.dt)*args.dt*(args.k').^2.*fft(ifft(Jinfo).*ifft(Jinfo).*ifft(dp));
    end
end
    