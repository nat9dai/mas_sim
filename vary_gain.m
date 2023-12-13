function main
    num_robot = 6;
    num_state = 3;
    num_input = 2;

    max_v = 2.0;
    max_w = 13.3;

    % adjacency matrix
    a = [0, 0, 1, 0, 0, 0;
         1, 0, 0, 0, 0, 1;
         1, 1, 0, 0, 0, 0;
         0, 1, 0, 0, 0, 0;
         0, 0, 1, 0, 0, 0;
         0, 0, 0, 1, 1, 0];

    dt = 0.01;
    T = 15;
    epsilon = 0.1;
    f = 1/dt;

    K_p_max = 6;
    K_h_max = 6;
    dK_p = 0.05;
    dK_h = 0.05;

    array_K_p = 0:dK_p:K_p_max;
    array_K_h = 0:dK_h:K_h_max;
    % array_t = zeros(length(array_K_p), length(array_K_h));
    array_V = zeros(length(array_K_p), length(array_K_h));

    for n_K_p = 1:length(array_K_p)
        K_p = array_K_p(n_K_p);
        for n_K_h = 1:length(array_K_h)
            K_h = array_K_h(n_K_h);

            % x = zeros(1, num_state*num_robot); % robots' states
            % initial position
            x = [-4.0, 2.0, 1.57, ...
                  0.0, -4.0, 1.0, ...
                  4.5, 2.1, 3.1, ...
                  4.0, -4.0, 2.0, ...
                 -1.0, -3.4, 1.7, ...
                 -4.0, -4.0, 1.2];

            p = reshape(x, num_state, []).';
            p = p(:, 1:2).';
            p = p(:).';

            u = zeros(1, num_input*num_robot); % robots' inputs

            % initial input u
            V = zeros(T*f, 1);
            for i = 1:num_robot
                where = find(a(i, :) == 1);
                ep = zeros(2, 1);
                for j_idx = 1:length(where)
                    j = where(j_idx);
                    ep = ep + [p(2*j-1) - p(2*i-1); p(2*j) - p(2*i)];
                end
                u(2*i-1) = sat(K_p*l2_norm(ep), max_v);
                u(2*i) = sat(K_h*wrap_to_pi(atan2(ep(2), ep(1)) - x(3*i)), max_w);
                V(1) = V(1) + dot(ep, ep);
            end

            t = 0;
            for k = 1:T*f
                u_reshaped = reshape(u, num_input, num_robot);
                x_new = zeros(size(x));
            
                % Update state for each robot
                for i = 1:num_robot
                    theta = x(3*i);
                    S_matrix = S(theta);
                    u_robot = u_reshaped(:, i);
                    x_new(1, (i-1)*num_state+1:i*num_state) = x((i-1)*num_state+1:i*num_state) + ...
                                                              (S_matrix * u_robot * dt)';
                end
                x = x_new;

                p = reshape(x, num_state, []).';
                p = p(:, 1:2).';
                p = p(:).';

                sum_ep = zeros(2, 1);

                % update input u
                for i = 1:num_robot
                    where = find(a(i, :) == 1);
                    ep = zeros(2, 1);
                    for j_idx = 1:length(where)
                        j = where(j_idx);
                        ep = ep + [p(2*j-1) - p(2*i-1); p(2*j) - p(2*i)];
                    end
                    u(2*i-1) = sat(K_p*l2_norm(ep), max_v);
                    u(2*i) = sat(K_h*wrap_to_pi(atan2(ep(2), ep(1)) - x(3*i)), max_w);
                    sum_ep = sum_ep + ep;
                    V(k) = V(k) + dot(ep, ep);
                end

                % if (l2_norm(sum_ep/num_robot) >= epsilon)
                %     t = t + dt;
                % else
                %     break;
                % end
            end
            % array_t(n_K_p, n_K_h) = t;
            array_V(n_K_p, n_K_h) = sat(mean(V(end-(100-1):end)), 30);
        end
        disp(n_K_p);
    end
    % Plotting
    [K_p_grid, K_h_grid] = meshgrid(array_K_p, array_K_h);
    figure;
    % surf(K_p_grid, K_h_grid, array_t');
    surf(K_p_grid, K_h_grid, array_V');
    % zlim([0 4]);
    xlabel('K_p');
    ylabel('K_h');
    % zlabel('Time (t)');
    zlabel('V(x_f)');
    title('3D Plot of V(x_f) vs K_p and K_h');
end

function S = S(theta)
    S = [cos(theta), 0; sin(theta), 0; 0, 1];
end

function Sbd = Sbd(thetas)
    Sbd = blkdiag(S(thetas(1)), S(thetas(2)), S(thetas(3)), S(thetas(4)), S(thetas(5)), S(thetas(6)));
end

function norm = l2_norm(vector)
    norm = sqrt(sum(vector.^2));
end

function angle = wrap_to_pi(angle)
    angle = mod(angle + pi, 2 * pi) - pi;
end

function u = sat(u, u_max)
    if abs(u) > u_max
        u = sign(u) * u_max;
    end
end
