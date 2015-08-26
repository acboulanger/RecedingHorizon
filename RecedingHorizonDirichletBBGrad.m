function [uconcat,yconcat,pconcat,Jinf,l2normY,l2normYTinf,bbiterations,args] = RecedingHorizonDirichletBBGrad()
    
    ClearClose();

    args = CreateParameters();
    args.matrices = BuildMatrices(args);

    % observation domain
    args.matrices.Obs = ...
        ComputeObservationMatrix(1,args.N+1,args);
    args.matrices.Adjoint = args.matrices.trial*...
        args.matrices.Obs*(args.matrices.trialTInv)';

    % control domain
    controldomain = zeros(1,args.N+1);
    %controldomain(1:end) = 1.0;
    controldomain(floor(1*(args.N+1)/12.0):floor(4*(args.N+1)/12.0)) = 1.0;
    controldomain(floor(8*(args.N+1)/12.0):floor(11*(args.N+1)/12.0)) = 1.0;
    [chi, chiT] = ...
        ComputeControlMatrix2(controldomain,args);
     args.matrices.B = chi;
     args.matrices.BT = chiT;
     
    %% Uncomment if you want to check gradient/hessian
 %    u = 1*ones(args.nmax+2, args.N+1);
 %    CheckGradientDirichlet(u, u, @solveState, @solveAdjoint, ...
 %    @compute_j, @compute_derivatives_j, args);
 %    CheckHessian(u, u, @solveState, @solveAdjoint, ...
 %    @solveTangent, @solveDFH, @compute_j, @compute_second_derivatives_j, args);
%      
     
     
     %% Check forward problem
      u = zeros(args.nmax+1,args.N+1);%initialization of the control
      args.kappa = 0.70;
      args.x0 = 0.0;
      args.y0 = 12*args.kappa^2*sech(args.kappa*(args.chebyGL - args.x0)).^2;%valeurs aux chebypoints
      
%       y = solveState(u,args);% one forward simulation for y
%       p = solveAdjoint(u,y,args);% one forward simulation for y
%       
%       plottedsteps=1:2:size(y.spatial,1);
%       [tg,xg] = meshgrid(args.tdata(plottedsteps),args.chebyGL(1:end));
%       
%        figure(1);
%        surf(xg,tg,y.spatial(plottedsteps,:)');
%        xlabel('x');ylabel('Time');zlabel('State variable y');
%        title('State Variable y');
%        view(-16,10);
%        shading interp;
%        
%        figure(2);
%        surf(xg,tg,p.spatial(plottedsteps,:)');
%        xlabel('x');ylabel('Time');zlabel('Adjoint variable p');
%        title('Adjoint Variable p');
%        view(-16,10);
%        shading interp;


%% Receding Horizon
    yconcat= zeros(floor(args.Tinf/args.dt)+1, args.N+1);
    pconcat= zeros(floor(args.Tinf/args.dt)+1, args.N+1);
    uconcat= zeros(floor(args.Tinf/args.dt)+1, args.N+1);
    Jinf = 0;
    l2normY = 0;
    bbiterations = 0;
    for irh=1:args.nrecinf
        if(irh>1)%initial condition is previous result at time = delta
            args.y0 = ykeep(end,:);
        end
        %run optimization process bis T;
        [y,p,u,jT,l2normyt,bbit,args] = solveOptimizationGlobal((irh-1)*args.deltarh,args);
        bbiterations = bbiterations + bbit;
        Jinf = Jinf + jT;
        l2normY = l2normY + l2normyt;
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
    end%end loop on deltas 
    l2normY = sqrt(l2normY);
    yspecend = args.matrices.trialT\(yconcat(end,:)');
    l2normYTinf = sqrt(yspecend'*(args.matrices.A*yspecend));%store last norm
    save('dirichletBBGradgamma01T2.mat');
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
    args.D = 5*pi; %domain is -50..50
    args.N = 256; %number of points
    args.k = args.N:-1:0;

    %Creation of Chebyshev Gauss-Lobatto points - our nodal basis
    args.chebyGL = cos(args.k*pi/args.N)*args.D;
    args.npoints = size(args.chebyGL,2);
    args.spacestep = [(args.chebyGL(2:end) - args.chebyGL(1:end-1))] ;
    args.ncells = args.npoints-1;
    
    %Receding horizon
    args.deltarh = 1.0;
    args.T = 2.0;
    args.Tinf = 200.0;
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



    % Optimization parameters
    args.gamma = 1e-1;
    args.tol = 1e-4;
    args.beta = 0.5;
    args.zeta = 0.001;
    
    
    %adaptive two-points method algo
    args.P = 40;% P >= 4M >= 8L
    args.M = 20;
    args.L = 5;%L <=5
    args.gam1 = args.M/args.L;%\geq 1, M/L
    args.gam2 = args.P/args.M;%\geq 1 P/M
    args.alphamin = 0.0;
    args.alphamax = 0.0;
    args.sigma1 = 0.1;% 0 < sigma1 < sigma2 < 1
    args.sigma2 = 0.9;
    args.LS = 1;
    

    % Misc
    args.coeffNL = 1.0;
    
    % default init
    args.y0 = zeros(1,args.N+1);
    args.dy0 = zeros(1,args.N+1);
    args.yobs = zeros(args.nmax+2,args.N+1);
    args.yspecobs = zeros(args.nmax+2,args.N-2)';
    
    args.normp = zeros(1,args.N+1);
    
    % physical parameters
    args.f =1.0;%-0.50;
    args.coeff3d = 1.0;%-1.0/6.0;
    args.coeffburgers = 1.0;%3.0/2.0;
    args.coeffsource = 1.0;%1.0/2.0;

end

function matrices = BuildMatrices(args)

    % Creation of Legendre polynomia
    LP = zeros(args.N+1, args.npoints);
    for i=0:args.N
    aux = legendre(i,args.chebyGL/args.D);
    LP(i+1,:) = aux(1,:);
    end

    % Creation of basis funtions for the dual petrov galerkin method - our spectral basis
    LP0 = LP(1:end-3,:);
    LP1 = LP(2:end-2,:);
    LP2 = LP(3:end-1,:);
    LP3 = LP(4:end,:);
    j = 0:(args.N-3);
    jm1 = 1:(args.N-2);
    jm2 = 2:(args.N-1);
    jm3 = 3:args.N;
    j1 = -1:(args.N-4);
    j2 = -2:(args.N-5);
    j3 = -3:(args.N-6);

    coeff = (2*j+3)./(2*j+5);
    diagcoeff = spdiags(coeff',0,args.N-2,args.N-2);
    LP1 = diagcoeff*LP1;
    LP3 = diagcoeff*LP3;
    trial = LP0 - LP1 - LP2 + LP3;
    test = LP0 + LP1 - LP2 - LP3;

    % Creation of the matrices

    % mass matrix
    mdiag0 = 2./(2*j+1) - 2*(2*j+3)./((2*j+5).^2) + 2./(2*j+5) - ...
        2*(2*j+3).^2./((2*j+7).*(2*j+5).*(2*j+5));
    mdiag1 = 6./(2*j1+7);
    mdiag2 = -2./(2*j2+5)+2*(2*j2+3)./((2*j2+5).*(2*j2+9));
    mdiag3 = -2*(2*j3+3)./((2*j3+5).*(2*j3+7));
    mdiagm1 = -6./(2*jm1+5);
    mdiagm2 = -2./(2*jm2+1) + 2*(2*jm2-1)./((2*jm2+1).*(2*jm2+5));
    mdiagm3 = 2*(2*jm3-3)./((2*jm3-1).*(2*jm3+1));
    M =args.D*spdiags([mdiagm3' mdiagm2' mdiagm1' mdiag0'...
        mdiag1' mdiag2' mdiag3'],...
        -3:3, args.N-2, args.N-2);

    % matrix for linear transport term 
    pdiag0 = 4*(2*j+3)./(2*j+5);
    pdiag1 = -8./(2*j1+7);
    pdiag2 = -2*(2*j2+3)./(2*j2+5);
    pdiagm1 = 8./(2*jm1+5);
    pdiagm2 = -2*(2*jm2-1)./(2*jm2+1);
    P = -spdiags([pdiagm2' pdiagm1' pdiag0' pdiag1' pdiag2'],...
        -2:2, args.N-2, args.N-2);

    % matrix for 3rd order term
    sdiag0 =2*(2*j+3).^2;
    S = 1.0/(args.D^2)*spdiags(sdiag0',0,args.N-2,args.N-2);

    % mass matrix for trial basis only (to compute adjoint)
    adiag0 = -2./(2*j+5) + 8./(2*j+7) + 2./(2*j+1);
    adiag1 = -2./(2*j1+5) + 2./(2*j1+7) - 2*(2*j1+3)./((2*j1+5).*(2*j1+7));
    adiag2 = -2./(2*j2+5) - 2*(2*j2+3)./((2*j2+5).*(2*j2+9));
    adiag3 = 2*(2*j3+3)./((2*j3+5).*(2*j3+7));
    adiagm1 = -2./(2*j+5) + 2./(2*j+7) - 2*(2*j+3)./((2*j+5).*(2*j+7));
    adiagm2 = -2./(2*j+5) - 2*(2*j+3)./((2*j+5).*(2*j+9));
    adiagm3 = 2*(2*j+3)./((2*j+5).*(2*j+7));
    A = args.D*spdiags([adiagm3' adiagm2' adiagm1' adiag0' adiag1' adiag2' adiag3'], ...
        -3:3, args.N-2, args.N-2);

    % Fill in the structure
    matrices.A = A;
    matrices.M = M;
    matrices.Msource = 0.5*M;
    matrices.MT = M';
    eps = 2*args.D/(args.N);
    matrices.Mreg = matrices.M + eps*eye(size(matrices.M,1));
    matrices.MTreg = matrices.MT + eps*eye(size(matrices.MT,1));
    matrices.MTInv = inv(M');
    matrices.S = args.coeff3d*S;
    matrices.P = P;
    matrices.PT = P';
    matrices.Pnl = args.coeffburgers*P;
    matrices.PnlT = (matrices.Pnl)';
    matrices.fP = args.f*P;
    matrices.fPT = (matrices.fP)';
    matrices.test = test;
    matrices.trial = trial;
    matrices.trialT = trial';
    matrices.testT = test';
    matrices.testTInv = pinv(test');
    matrices.trialT = trial';
    matrices.trialTInv = pinv(trial');
    matrices.left=0.5*(M +args.dt*S);
    matrices.leftInv = inv(matrices.left);
    matrices.leftTInv = inv(matrices.left');
    matrices.right=0.5*(M -args.dt*S);
    matrices.rightT=matrices.right';
    matrices.M_leftTinv_dt = matrices.leftTInv*args.dt;
    matrices.M_leftTinv_rightT = matrices.leftTInv*(matrices.rightT);
    
    % control is discretized with linear finite elements in space
    % piecewise constant in time
    % construct the lumped mass matrix for space
    dx = [0,args.spacestep,0];
    matrices.MassS = spdiags(0.5*(dx(2:end) + dx(1:(end-1)))',...
        0,args.N+1, args.N+1);
    %matrices.MassSSource = 0.5*matrices.MassS;
    % construct the lumped mass matrix for time (actually P0 in time)
    matrices.MassT = args.dt*speye(args.nmax+1);
    matrices.MassT(1,1) = 0.5*args.dt;
    matrices.MassT(end,end) = 0.5*args.dt;
    % construct the matrix A for the inner product in H (space and time)
    % (u,v)_H = u'Av
    matrices.Mass = kron(matrices.MassS,matrices.MassT);%should be diag(MassT(i)*MassS)
    %matrices.MassSource = kron(matrices.MassSSource,matrices.MassT);
    
end

function [Obs] = ComputeObservationMatrix(i1,i2,args)
    observationdomain = i1:i2;
    Obs = zeros(args.N+1);
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
    B = zeros(args.N+1);
    for i=1:(args.N+1)
        if(controldomain(i)==1)
            B(i, i) = 1.0;
        end
    end
BT = B';  
end

%% %%%%%%%%%%%%%%%% solveState functions %%%%%%%%%%%%%%%%
function y = solveState(u, args)
    % parameters
    dt=args.dt;
    nmax=args.nmax;
    N=args.N;
    coeffNL=args.coeffNL;
    matrices = args.matrices;
    
    % state variables
    y.spatial = zeros(nmax+2,N+1);% spatial domain
    y.spec = zeros(nmax+2,N-2);% spectral domain
    
    % initialization
    yspec0 = matrices.trialT\(args.y0)';
    y.spatial(1,:) = args.y0;
    y.spec(1,:) = yspec0;
    u = ((args.matrices.B)*(u'))';%effect of indicator function

    % first time step in the spectral space, semi implicit
    NLterm = (args.y0).^2;
    pNLterm=coeffNL*matrices.trialTInv*NLterm';
    yspec1 = matrices.leftInv*((0.5*matrices.M)*yspec0 ...
            + 0.5*dt*(-0.5*matrices.Pnl*pNLterm + matrices.fP*yspec0...
            + matrices.test*matrices.MassS*u(1,:)'));
    y1 = matrices.trialT*yspec1;
    y.spatial(2,:) = y1;
    y.spec(2,:) = yspec1;

    % Time loop for interior steps - Crank Nicolson Leap Frog scheme
    ym1 = y1;
    yspecm1 = yspec1;
    yspecm2 = yspec0;
    for i=2:nmax
        NLterm = ym1.*ym1;
        pNLterm=coeffNL*matrices.trialTInv*NLterm;
        yspeci = (matrices.leftInv)*( (matrices.right)*yspecm2 ...
          + dt*(-0.5*matrices.Pnl*pNLterm...
          + matrices.fP*yspecm1 ...
          + matrices.test*matrices.MassS*u(i,:)') );
        yi = matrices.trialT*yspeci;
        ym1 = yi;
        yspecm2 = yspecm1;
        yspecm1 = yspeci;
        y.spec(i+1,:) = yspeci;
        y.spatial(i+1,:) = yi;
       % if (sum(isnan(yi))>0)
       %     fprintf('stop NAN');
       % end
    end

    %last step 
    %minv=inv(matrices.M);
    NLterm = ym1.^2;
    pNLterm=coeffNL*matrices.trialTInv*NLterm;
    
%     rhs = ((matrices.right)*yspecm2 +...
%                 0.5*dt*(0.5*matrices.P*pNLterm + matrices.P*yspecm1)...
%                 + 0.5*matrices.M*yspecm1...
%                 + 1.0*0.5*dt*matrices.M*qspec(nmax+1,:)');
%     [yspecend,success,residual,itermeth] = gmres(matrices.M,rhs,[],args.tolgmres,N-2);
    yspecend = (matrices.Mreg)\((matrices.right)*yspecm2 ...
                + 0.5*dt*(-0.5*matrices.Pnl*pNLterm + matrices.fP*yspecm1)...
                + 0.5*matrices.M*yspecm1...
                + 1.0*0.5*dt*matrices.test*matrices.MassS*u(nmax+1,:)');
    yend = matrices.trialT*yspecend;
    y.spec(end,:) = yspecend;
    y.spatial(end,:) = yend;
    
end


%% %%%%%%%%%%%%%%%% solveAdjoint functions %%%%%%%%%%%%%%%%
function p = solveAdjoint(u,y,args)
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

    % initialization 
    % M is ill conditionned. Regularization added
    mt = matrices.MTreg;
    %rhs = (y.spec(end,:)' - args.yspecobs);
    %[pspec0,success,residual,itermeth] = gmres(mt,rhs,[],args.tolgmres,N-2);
    rhsspatial = args.matrices.Obs*(y.spatial(end,:)'- args.yobs(end,:)');
    rhs = matrices.trialT\rhsspatial;
    %pspec0 = -mt\(matrices.A*(y.spec(end,:)' - args.yspecobs));
    pspec0 = -mt\(dt*matrices.Adjoint*matrices.A*rhs);%%%%% add the dt
    p0 = (matrices.testT)*(pspec0);
   
    p.spatial(1,:) = p0;
    p.spec(1,:) = pspec0;

    % first step
    NLterm = 0.5*2.0*yrev(2,:)'.*...
        (matrices.trialTInv'*(matrices.PnlT*pspec0));
    pNLterm=coeffNL*matrices.trial*NLterm;
    rhsspatial = args.matrices.Obs*(yrev(2,:)'- yobsrev(2,:)');
    rhs = matrices.trialT\rhsspatial;
    pspec1 = matrices.leftTInv*(0.5*matrices.MT*pspec0 +...
        0.5*dt*(matrices.fPT*pspec0 - pNLterm...
        -2.0*matrices.Adjoint*matrices.A*rhs));
    p1 = matrices.testT*pspec1;
    p.spatial(2,:) = p1;
    p.spec(2,:) = pspec1;
    
    pspecm1 = pspec1;
    pspecm2 = pspec0;
    %Time loop
    for i = 2:nmax
        NLterm = 0.5*2.0*yrev(i+1,:)'.*...
            (matrices.trialTInv'*(matrices.PnlT*pspecm1));
        pNLterm=coeffNL*matrices.trial*NLterm;
        rhsspatial = args.matrices.Obs*(yrev(i+1,:)'- yobsrev(i+1,:)');
        rhs = matrices.trialT\rhsspatial;
        pspeci = matrices.M_leftTinv_rightT*pspecm2...
            + matrices.M_leftTinv_dt* (matrices.fPT*pspecm1 - pNLterm...
            - matrices.Adjoint*matrices.A*rhs);
        pi = matrices.testT*pspeci;
        pspecm2 = pspecm1;
        pspecm1 = pspeci; 
        p.spec(i+1,:) = pspeci;
        p.spatial(i+1,:) = pi;
    end
    p.spec = p.spec(end:-1:1,:);
    p.spatial = p.spatial(end:-1:1,:);
end

%% %%%%%%%%%%%%%%%% Tracking term %%%%%%%%%%%%%%%%

function l2normy = compute_l2normy(u,y,args)
    l2normy = 0;
    for i=1:args.nkeep
        discr = args.matrices.Obs*(y.spatial(i,:)'-args.yobs(i,:)');
        ymyd = args.matrices.trialT\discr;
        l2normy = l2normy+ args.dt*ymyd'*(args.matrices.A*ymyd);
    end
end

function [j,jkeep] = compute_j(u,y,args)
    j = 0;
    for i=2:args.nkeep
        discr = args.matrices.Obs*(y.spatial(i,:)'-args.yobs(i,:)');
        ymyd = args.matrices.trialT\discr;
        j = j+ 0.5*args.dt*ymyd'*(args.matrices.A*ymyd) + 0.5*args.gamma*args.dt*u(:)'*(args.matrices.Mass*u(:));
    end
    jkeep=j;
    for i=(args.nkeep+1):args.nmax+2
        discr = args.matrices.Obs*(y.spatial(i,:)'-args.yobs(i,:)');
        ymyd = args.matrices.trialT\discr;
        j = j+ 0.5*args.dt*ymyd'*(args.matrices.A*ymyd) + 0.5*args.gamma*args.dt*u(:)'*(args.matrices.Mass*u(:));
    end
end

function [dj,dj2] = compute_derivatives_j(u,y,p,args)%do not forget time in inner product
    %p: row = time, column = space
    dj2 = -((args.matrices.BT)*(p)')' + args.gamma*u;%each column is B*p(t_i)
    dj = dj2(:);%makes a vector
end

function [jnew,unew,ynew,jref,jmin,jc,jmax,jarray,l,p,jnewkeep,l2normy] = AdaptiveTwoPointsNonmonotonLS(u,gradcol,gradmat,ngrad,... 
                                            jmax,jmin,jref,jc,jold,jarray,stepsize,l,p,args)
    
    %compute new iterate
    unew = u - stepsize*gradmat;
    ynew = solveState(unew,args);
    [jnew,jnewkeep] = compute_j(unew,ynew,args);
    
    if (args.LS)
        if(l==args.L)%max authorized growing steps reached
            if ( (jmax-jmin)/(jc-jmin) > args.gam1)
                jref = jc;
            else
                jref = jmax; 
            end
            l = 0;
        end

        if ( (p > args.P) && (jmax > jold) && ((jref-jold)/(jmax-jold)>= args.gam2) )
            jref = jmax;  
        end

        m = 0;
        if((jnew - jold) <= -args.zeta*stepsize*ngrad*ngrad)%test armijo rule
            p = p+1;
        else
            p=0;
            m=0;
            while( (jnew - min(jref,jmax)) > -args.zeta*stepsize*ngrad*ngrad && m < 5)
                %jnew - min(jref,jmax)
                m = m+1
                stepsize = stepsize*args.beta;
                unew = u - stepsize*gradmat;
                ynew = solveState(unew,args);
                [jnew,jnewkeep] = compute_j(unew,ynew,args);
            end
        end
        
        l2normy = compute_l2normy(unew,ynew,args);
        %update different values for j
        if(jnew < jmin)%current iterate is the best
            jc = jnew;
            jmin = jnew;
            l = 0;
        else
            l = l+1;
        end

        if(jnew > jc)%current iterate is bigger than ref
            jc = jnew;
        end

        %update jmax
        jsize = size(jarray,2);
        if jsize == args.M
            jarray = [jarray(2:end),jnew];
        else
            jarray = [jarray,jnew];
        end
        jmax = max(jarray);
    end
    if(args.LS==0)
       l2normy = compute_l2normy(unew,ynew,args);
    end
end

function [y,p,u,jkeep,l2normyt,bbit,args] = solveOptimizationGlobal(t0,args)

    %params and init
    gamma = args.gamma;
    M = args.matrices.Mass;
    
    %%  Start of Gradient descent strategy
    fprintf('Gradient descent strategy...\n');
    fprintf('gamma = %d , t_0 = %d, delta = %d\n', ...
        gamma, t0, args.deltarh);
    
    uold = zeros(args.nmax+1, args.N+1);%initial guess spectral space
    
    yold = solveState(uold,args);% one forward simulation for y
    [jold,jkeepold] = compute_j(uold,yold,args)
    pold = solveAdjoint(uold,yold,args);% one forward simulation for y

    [grad,grad2] = compute_derivatives_j(uold,yold.spatial,pold.spatial,args);
    gradold = grad;
    gradold2=grad2;
    ngrad = sqrt(grad'*M*grad)
    
    
    lindex = 0;
    pindex = 0;
    jmin = jold;
    jref = jold;
    jc = jold;
    jarray = [jold];
    jmax = jold;
    
    bbit = 0;
    while(ngrad > args.tol && bbit < 500)
        
        if(bbit==0)
           stepsize = 1.0/ngrad;
        end
        
        [j,u,y,jref,jmin,jc,jmax,jarray,l,p,jkeep,l2normyt] = AdaptiveTwoPointsNonmonotonLS(uold,gradold,gradold2,ngrad,... 
                                            jmax,jmin,jref,jc,jold,jarray,stepsize,lindex,pindex,args);
        p = solveAdjoint(u,y,args);% one forward simulation for y
        %myvisu(1,y.spatial,p.spatial,uspace,args)

        [grad,grad2] = compute_derivatives_j(u,y.spatial,p.spatial,args);
        ngrad = sqrt(real(grad'*M*grad))
        
        %stepsize
        s = u - uold;
        s = s(:);
        g = grad - gradold;%already in column
        stg = real(g'*M*s);
        if(stg > 0)
            %disp('stg > 0');
            if mod(bbit,2)
                sts = real(s'*M*s);
                stepsize = sts/stg;
            else
                gtg = real(g'*M*g);
                stepsize = stg/gtg;
            end
        else
            %disp('stg < 0');
            stepsize = 1.0/ngrad;
        end
        
        uold = u;
        jold = j;
        gradold = grad;
        gradold2 = grad2;
        
        bbit = bbit+1
        j
    end
    if(bbit==0)
        y = yold;
        p = pold;
        u = uold;
        jkeep = jkeepold;
        l2normyt = compute_l2normy(u,y,args);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% VISUALIZATION %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function visunormq(nfig,q,gamma,args)
    q2 = reshape(q,args.nmax+1,args.N+1);
    L2NormInTimeQ = sqrt(sum(args.matrices.MassT*((q2).*(q2))));
    figure(nfig);
    clf(nfig);
    hold on;
    plot(L2NormInTimeQ);
    plot(args.alpha/gamma*ones(1,size(L2NormInTimeQ,2)));
    xlabel('x');ylabel('||q||_{L^2(I)}');
    hold off;
end

function visunormL2(nfig,y,args)
    n = size(y,1);
    L2NormInSpace = zeros(1,n);
    for i=1:n
        discr = y(i,:)';
        ymyd = args.matrices.trialT\discr;
        L2NormInSpace(i) = ymyd'*(args.matrices.A*ymyd);
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
    [tg,xg] = meshgrid(args.tdatarh(plottedsteps),args.chebyGL(1:end));
    
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