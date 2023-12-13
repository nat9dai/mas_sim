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
    T = 10;
    epsilon = 0.1;
    f = 1/dt;

    K_p = 4.0;
    K_h = 4.0;

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
    
    V = zeros(T*f, 1);
    % initial input u
    for i = 1:num_robot
        where = find(a(i, :) == 1);
        ep = zeros(2, 1);
        for j_idx = 1:length(where)
            j = where(j_idx);
            ep = ep + [p(2*j-1) - p(2*i-1); p(2*j) - p(2*i)];
        end
        u(num_input*i-1) = sat(K_p*l2_norm(ep), max_v);
        u(num_input*i) = sat(K_h*wrap_to_pi(atan2(ep(2), ep(1)) - x(3*i)), max_w);
        V(1) = V(1) + dot(ep, ep);
    end

    % initialize trajectories
    traj_x = zeros(T*f, num_state*num_robot);
    traj_p = zeros(T*f, 2*num_robot);
    traj_x(1, :) = x;
    traj_p(1, :) = p;

    t = 0;
    for k = 2:T*f
        u_reshaped = reshape(u, num_input, num_robot);
        x_new = zeros(size(x));

        % Update state for each robot
        for i = 1:num_robot
            theta = x(3*i);
            S_matrix = S(theta);
            u_robot = u_reshaped(:, i);

            % Update the state
            x_new(1, (i-1)*num_state+1:i*num_state) = x((i-1)*num_state+1:i*num_state) + ...
                (S_matrix * u_robot * dt)';
        end
        traj_x(k, :) = x_new;
        x = x_new;

        p = reshape(x, num_state, []).';
        p = p(:, 1:2).';
        p = p(:).';
        traj_p(k, :) = p;

        sum_ep = zeros(2, 1);

        % Update input u
        for i = 1:num_robot
            where = find(a(i, :) == 1);
            ep = zeros(2, 1);
            for j_idx = 1:length(where)
                j = where(j_idx);
                % Adjusted indexing
                ep = ep + [p(2*j-1) - p(2*i-1); p(2*j) - p(2*i)];
            end
            u(num_input*i-1) = sat(K_p*l2_norm(ep), max_v);
            u(num_input*i) = sat(K_h*wrap_to_pi(atan2(ep(2), ep(1)) - x(3*i)), max_w);
            sum_ep = sum_ep + ep;
            V(k) = V(k) + dot(ep, ep);
        end

        if (l2_norm(sum_ep/num_robot) >= epsilon)
            t = t + dt;
        end
    end

    disp(t)

    % Plotting
    p_x = traj_p(:, 1:2:end);  % even columns are x positions
    p_y = traj_p(:, 2:2:end);  % odd columns are y positions
    
    time = linspace(0, T, T*f);
    
    figure;
    subplot(3, 1, 1);
    hold on;
    for i = 1:num_robot
        plot(time, p_x(:, i), 'DisplayName', sprintf('Robot %d p_x', i));
    end
    title('p_x vs Time');
    xlabel('Time (seconds)');
    ylabel('p_x');
    legend;
    hold off;
    grid;
    
    subplot(3, 1, 2);
    hold on;
    for i = 1:num_robot
        plot(time, p_y(:, i), 'DisplayName', sprintf('Robot %d p_y', i));
    end
    title('p_y vs Time');
    xlabel('Time (seconds)');
    ylabel('p_y');
    legend;
    hold off;
    grid;

    subplot(3, 1, 3);
    hold on;
    plot(time, V/2, 'DisplayName', sprintf('V(x)'));
    title('V(x) vs Time');
    xlabel('Time (seconds)');
    ylabel('V(x)');
    legend;
    hold off;
    grid;
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
