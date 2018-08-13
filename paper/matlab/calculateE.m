%solving for E
for i = 1:4
    for j = 2:5
        comm(i,j) = 2*a(i,j)*b(i,j) + a(i,j)^2;
        vol(i,j) = b(i,j)* a(i,j)^2;
        numDomains(i,j) = M*N*K / vol(i,j);
        commTotal(i,j) = comm(i,j) * numDomains(i,j);
        E(i,j) = simplify(optimal / commTotal(i,j));
    end
end

P1 = M*N/S;
P2 = M*N*K/S^(3/2);
P3 = M*N*K/(S/3)^(3/2);
P = [1;P1;P2;P3];
%getting P values
for i = 1:4
    Pval(i) =  eval(subs(P(i),[M N K S], [MM NN KK SS]));
end

%getting P ranges
for i = 1:3
    PP{i} = PPP(PPP >= Pval(i) & PPP < Pval(i+1));
end
PP{4} = PPP(PPP >= Pval(4));


EE = cell(1,4);
for i = 1:4
    for j = 2:5
        EE{j-1} = [EE{j-1} eval(subs(subs(E(i,j),[M N K S], [MM NN KK SS]), p, PP{i}))];
    end
end

plot(PPP, EE{1}, PPP, EE{2}, PPP, EE{3}, PPP, EE{4});
y = 0:0.1:1;
hold on
for i = 2:4
    x{i} = Pval(i) * ones(1,length(y));
    plot(x{i},y, 'k--');
    
    txt1 = strcat('p', num2str(i-1),' = ', num2str(floor(Pval(i))));
    x1 = Pval(i);
    if (i == 2)
        x1 = x1 + 6000;
    else
        x1 = x1 + 2000;
    end
    text(x1,0.5 - 0.1*i,txt1);
end
hold off
title(strcat('M = ', num2str(MM), ', N = ', num2str(NN), ', K = ', num2str(KK), ...
    ' S = ', num2str(SS)));
xlabel('p');
ylabel('E(p)');
legend('ij', 'ijk', 'cubic', 'optimal');