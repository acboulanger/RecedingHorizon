function CheckGradient(u, du, solveState, solveAdjoint, compute_j, compute_derivatives_j, args)

y = solveState(u, args);
fprintf('State ok \n')
z = solveAdjoint(u, y, args);
fprintf('Adjoint ok \n')
g = compute_derivatives_j(u, y, z, args);
fprintf('Derivative ok \n')
%dup = du(1:end-1,:);% last step does not count - adjoint has nmax +1 steps while state has nmax+2
%du = ((args.matrices.B)*(u)')';
fftdu = fft(du,[],2);
jprime = g'*fftdu(:);
%jprime = sum(g.*fftdu(:));

for i = -2:10
    epsilon = sqrt(10)^(-(i-1));
    
    up = u + epsilon*du;
    yp = solveState(up, args);
    jp = compute_j(up, yp, args);
    %fprintf('jp ok \n')

    um = u - epsilon*du; 
    ym = solveState(um, args);
    jm = compute_j(um, ym, args);
    %fprintf('jm ok \n')
    
    %jprime = g'*dq;
    jdiff = 0.5*(jp - jm) / epsilon;
    rerr(i+3) = abs(jprime - jdiff) / abs(jprime);
    fprintf('jp: %f, difference quot.: %f, rel. err.: %e\n', jprime, jdiff, rerr(i+3));
end

semilogy(rerr);

end
