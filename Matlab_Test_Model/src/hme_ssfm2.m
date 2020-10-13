%%
g0 = 0.4; PsatTR = 1; loss = 0.2;
Omega = sqrt(10); del = 0.03;
gam = 4; beta2 = -2;
test = hme_ssfm2(g0, PsatTR, loss, Omega, del, gam, beta2)
%%
function output = hme_ssfm2(g0, PsatTR, loss, Omega, del, gam, beta2)
    %equation parameters
%     g0 = 0.4; PsatTR = 1; loss = 0.2;
%     Omega = sqrt(10); del = 0.03;
%     gam = 4; beta2 = -2;
    % Discretization
    Nt = 1024; T = 50; dt = T/Nt;
    t = (-Nt/2:1:Nt/2 - 1)' * dt;
    dw = 2 * pi/T; w = [0:Nt/2-1 0 -Nt/2 + 1: -1 ]' * dw;
    Z = 400; h = 0.04; NumSteps = round(Z/h);
    SaveInterval = 250;
    % Operators
    L = (1i * beta2 * w.^2 - loss)/2;
    K = (1 - (w/Omega).^2)/2;
    % Initial condition
    u0 = 0.25 * exp(-(t/5).^2);
    uf = fft(u0); uplot = abs(u0).';
    zplot = 0; Psatf = PsatTR/dt * Nt;
    for istep = 1:NumSteps
        g1 = g0/(1 + norm(uf)^2/Psatf);
        g2 = -2 * (g1^2/g0/Psatf) *...
            real(dot(uf,(L + g1 * K).* uf));
        u = ifft(exp(L * h/2 + (g1 * h/2 + g2/8 * h^2) * K).* uf);
        uf = fft(exp(-(del + 1i * gam)/(2 * del)...
        * log(1 - 2 * del * h * abs(u).^2)).* u);
        g1 = g0/(1 + norm(uf)^2/Psatf);
        uf = exp(L * h/2 + (g1 * h/2 + g2/8 * h^2) * K).* uf;
        if mod(istep, SaveInterval) ==0
            uplot = [uplot; abs(ifft(uf)).'];
            zplot = [zplot; istep * h];
        end
    end
    waterfall(t, zplot, uplot); colormap([0 0 0]);
    view(30, 30); xlabel('t'); ylabel('z');
    output = 1;
end

