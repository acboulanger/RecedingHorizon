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
jprime = g'*args.Mass*fftdu(:);
%jprime = sum(g.*fftdu(:))

for i = 1:10
    epsilon = sqrt(10)^(-(i-1));
    
    up = u + epsilon*du;
    yp = solveState(up, args);
    jp = compute_j(up, yp, args)
    %fprintf('jp ok \n')

    um = u - epsilon*du; 
    ym = solveState(um, args);
    jm = compute_j(um, ym, args)
    %fprintf('jm ok \n')
    
    figure(1);
    hold on;
    plot(yp.spatial(end,:));
    plot(ym.spatial(end,:));
    drawnow()
    hold off;
    
    jdiff = 0.5*(jp - jm) / epsilon;
    rerr(i) = abs(jprime - jdiff) / abs(jprime);
    fprintf('jp: %f, difference quot.: %f, rel. err.: %e\n', jprime, jdiff, rerr(i));
end

semilogy(rerr);

end
