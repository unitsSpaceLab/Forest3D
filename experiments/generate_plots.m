%%% Generate plots used for IFIT2026 paper
% Switch 'base_dir' at the beginning of "Load" section between uphill and
% downhill case studies, than run the script.
% 

clear all;
clc;

%% Load
base_dir = "data_uphill";
% base_dir = "data_downhill";

folds_suffix = {'LUNA','FOREST','GAZEBO'}';

folds = cell(length(folds_suffix),1);
folds_all = dir(base_dir);
for ii = 1:length(folds_suffix)
    folds_indx = find(arrayfun(@(x) endsWith(x.name,folds_suffix{ii}), folds_all));
    if length(folds_indx) > 1
        error('More than one folders end with suffix [%s]', folds_suffix{ii});
    end
    folds{ii} = folds_all(folds_indx).name;
end

stl = stlread(fullfile(base_dir,"terrain_cropped_run.stl"));

wheels = {'bl', 'br', 'fl', 'fr'}';
res_odom = cell(length(folds), 1);
res_inter_forces = cell(length(folds), length(wheels));
res_inter_states = cell(length(folds), length(wheels));
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
for ii = 1:length(folds)
    % ----- odometry -----
    if exist(fullfile(base_dir,folds{ii},'csvs','odom.csv'), "file")
        res_odom{ii,1} = readtable(fullfile(base_dir,folds{ii},'csvs','odom.csv'));
    elseif exist(fullfile(base_dir,folds{ii},'topic_csvs','_Archimede_odom.csv'), "file")
        res_odom{ii,1} = readtable(fullfile(base_dir,folds{ii},'topic_csvs','_Archimede_odom.csv'));
        res_odom{ii,1} = res_odom{ii,1}(:,[1,6:12,49:54]);
        res_odom{ii,1}.Properties.VariableNames = {'time','pos_x','pos_y','pos_z', ...
            'quat_x','quat_y','quat_z','quat_w','vel_x','vel_y','vel_z','ang_vel_x','ang_vel_y','ang_vel_z'};
        % res_odom{ii,1}.time = 1e-9 * (res_odom{ii,1}.time - res_odom{ii,1}.time(1));
    end

    % ----- interaction forces -----
    if strcmp(folds_suffix{ii}, 'GAZEBO')
        for jj = 1:length(wheels)
            res_inter_forces{ii,jj} = readtable(fullfile(base_dir,folds{ii},'topic_csvs',...
                strcat('_Archimede_',wheels{jj},'_wheel_contact.csv')));
            res_inter_forces{ii,jj} = res_inter_forces{ii,jj}(:,[1,14:16,25:30]);
            res_inter_forces{ii,jj}.Properties.VariableNames = ...
                {'time','contact_normal_x','contact_normal_y','contact_normal_z', ...
                 'Fx_world','Fy_world','Fz_world','Mx_world','My_world','Mz_world'};
            % res_inter_forces{ii,jj}.time = 1e-9 * (res_inter_forces{ii,jj}.time - res_inter_forces{ii,jj}.time(1));
        end
    else
        if exist(fullfile(base_dir,folds{ii},'csvs','terra_forces.csv'), "file")
            tmp_tab = readtable(fullfile(base_dir,folds{ii},'csvs','terra_forces.csv'));
            % tmp_tab.time = tmp_tab.time-tmp_tab.time(1);
            for jj = 1:length(wheels)
                tmp_indx = string(tmp_tab.wheel) == wheels{jj};
                res_inter_forces{ii,jj} = tmp_tab(tmp_indx,:);
            end
        else
            for jj = 1:length(wheels)
                res_inter_forces{ii,jj} = readtable(fullfile(base_dir,folds{ii},'topic_csvs',...
                    strcat('_Archimede_',wheels{jj},'_wheel_link_terramech_forces.csv')));
                res_inter_forces{ii,jj}.Properties.VariableNames = ...
                    {'time','Fx_contact','Fy_contact','Fz_contact','Mx_contact','My_contact','Mz_contact',...
                     'Fx_world','Fy_world','Fz_world','Mx_world','My_world','Mz_world'};
                % res_inter_forces{ii,jj}.time = 1e-9 * (res_inter_forces{ii,jj}.time - res_inter_forces{ii,jj}.time(1));
            end
        end
    end

    % ----- interaction states -----
    if exist(fullfile(base_dir,folds{ii},'csvs','terra_state.csv'), "file")
        tmp_tab = readtable(fullfile(base_dir,folds{ii},'csvs','terra_state.csv'));
        % tmp_tab.time = tmp_tab.time-tmp_tab.time(1);
        for jj = 1:length(wheels)
            tmp_indx = string(tmp_tab.wheel) == wheels{jj};
            res_inter_states{ii,jj} = tmp_tab(tmp_indx,:);
        end
    else
        for jj = 1:length(wheels)
            res_inter_states{ii,jj} = readtable(fullfile(base_dir,folds{ii},'topic_csvs',...
                strcat('_Archimede_',wheels{jj},'_wheel_link_terramech_state.csv')));
            res_inter_states{ii,jj}.Properties.VariableNames = ...
                {'time','omega','v_x','v_y','slip_ratio','slip_angle_deg','wheel_load','sinkage'};
            % res_inter_states{ii,jj}.time = 1e-9 * (res_inter_states{ii,jj}.time - res_inter_states{ii,jj}.time(1));
        end
    end
    
    % ----- manage timestamps -----
    time_start = min([res_odom{ii}.time(1) ...
        cellfun(@(x) x.time(1), res_inter_forces(ii,:)) ...
        cellfun(@(x) x.time(1), res_inter_states(ii,:))]);
    res_odom{ii}.time = 1e-9 * (res_odom{ii}.time - time_start);
    for jj = 1:length(wheels)
        res_inter_forces{ii,jj}.time = 1e-9 * (res_inter_forces{ii,jj}.time - time_start);
        res_inter_states{ii,jj}.time = 1e-9 * (res_inter_states{ii,jj}.time - time_start);
    end

end
warning('on','MATLAB:table:ModifiedAndSavedVarnames');

clear folds_all folds_indx tmp_tab tmp_indx

%% Average wheels slip
% Not all wheels are always in contact, so average on the ones that are in
% contact at the current timestamp
func_smooth = @(x) movmedian(x,1);

res_slips = cell(size(res_inter_states,1), 1);

for ii = 1:size(res_inter_states,1)
    % all timestamps
    timestamps = unique(cell2mat(cellfun(@(x) x.time, res_inter_states(ii,:)','UniformOutput',false)));
    timestamps_N = zeros(length(timestamps),1);
    slip_ratio = zeros(length(timestamps),1);
    slip_angle_deg = zeros(length(timestamps),1);
    for jj = 1:size(res_inter_states,2)
        [tmp_indx,tmp_loc] = ismember(timestamps, res_inter_states{ii,jj}.time);
        tmp_loc(tmp_loc == 0) = [];
        slip_ratio(tmp_indx) = slip_ratio(tmp_indx) + func_smooth(res_inter_states{ii,jj}.slip_ratio(tmp_loc));
        slip_angle_deg(tmp_indx) = slip_angle_deg(tmp_indx) + func_smooth(res_inter_states{ii,jj}.slip_angle_deg(tmp_loc));
        timestamps_N(tmp_indx) = timestamps_N(tmp_indx) + 1;
    end
    res_slips{ii}.time = timestamps;
    res_slips{ii}.slip_ratio = slip_ratio ./ timestamps_N;
    res_slips{ii}.slip_angle_deg = slip_angle_deg ./ timestamps_N;
end

clear func_smooth timestamps timestamps_N slip_ratio slip_angle_deg tmp_indx tmp_loc

%% Plot 3D trajectories - paper
stop_time = 25; % empty or >end_time to get all data
include_slips_patch = false;
include_rover_frames = false;
rover_frames_samples = 10;

figure;
trisurf(stl, 'edgealpha',0.3, 'facealpha',1, 'edgecolor','k');%'interp');
% trimesh(stl);
axis equal;
hold on;
cc = [     ...0 0.4470 0.7410; ...
       0.39   0.83   0.07; ...
     0.9290 0.6940 0.1250; ...
     0.8500 0.3250 0.0980; ...
     0.4940 0.1840 0.5560; ...
          1      0      1; ...
          0      1      0];
% c = ('rgb')';
for ii = 1:length(res_odom)
    if ~isempty(stop_time) && res_odom{ii}.time(end) > stop_time
        stop_indx = find(res_odom{ii}.time >= stop_time, 1,"first");
    else
        stop_indx = length(res_odom{ii}.time);
    end

    plot3(res_odom{ii}.pos_x(1:stop_indx), res_odom{ii}.pos_y(1:stop_indx), res_odom{ii}.pos_z(1:stop_indx)-0.14, ...
        'Color',cc(ii,:), 'LineWidth',3);

    if include_rover_frames
        fr_indx = round(linspace(1,stop_indx,rover_frames_samples));
        plotTransforms([res_odom{ii}.pos_x(fr_indx) res_odom{ii}.pos_y(fr_indx) res_odom{ii}.pos_z(fr_indx)] - ones(length(fr_indx),3).*[0 0 0.14], ...
            [res_odom{ii}.quat_w(fr_indx) res_odom{ii}.quat_x(fr_indx) res_odom{ii}.quat_y(fr_indx) res_odom{ii}.quat_z(fr_indx)], ...
            'FrameSize',0.5);
    end

    % add slip curves
    if include_slips_patch && exist("res_slips", "var")
        slip1scale = 1;
        base_offset = 0; %slip1scale;
        % smooth with movmedian and interpolate to the odom timestamps
        slip = interp1(res_slips{ii}.time, movmedian(res_slips{ii}.slip_ratio,100), res_odom{ii}.time(1:stop_indx));
        % scale
        slip = slip * slip1scale;
        % plot baseline and slip
        patch_z = base_offset + [min(slip, 0); flip(max(slip, 0))];
        patch([res_odom{ii}.pos_x(1:stop_indx); flip(res_odom{ii}.pos_x(1:stop_indx))], ...
              [res_odom{ii}.pos_y(1:stop_indx); flip(res_odom{ii}.pos_y(1:stop_indx))], ...
              [res_odom{ii}.pos_z(1:stop_indx); flip(res_odom{ii}.pos_z(1:stop_indx))] + patch_z, ...
              cc(ii,:), 'FaceAlpha',0.5);
        plot3(res_odom{ii}.pos_x(1:stop_indx), res_odom{ii}.pos_y(1:stop_indx), ...
            res_odom{ii}.pos_z(1:stop_indx) + base_offset, '--','Color',cc(ii,:));
    end

    % view(-175, 6);
    view(182, 1);
end
cb = colorbar;
cb.TickLabels = string(double(string(cb.TickLabels)) - double(string(cb.TickLabels(1))));
cb.Label.String = 'Elevation [m]';
cb.FontSize = 12;
xticklabels([]); yticklabels([]); zticklabels([]);
box;
gca().set("TickLength",[0 0]);
legend([{""}; folds_suffix], 'FontSize',14);
colormap('bone')

clear stop_time include_slips_patch include_rover_frames rover_frames_samples cc stop_indx fr_indx ...
    slip1scale base_offset slip patch_z

%% Plot slips - paper
figure;
tiledlayout(2,1, 'TileSpacing','tight');
% c = [0.9290 0.6940 0.1250; ...
%           1      0      1; ...
%           0      1      0];
func = @(x) movmedian(x,100);

nexttile(1);
hold on;
% cellfun(@(x,cc) plot(x.time, x.slip_ratio, 'Color',cc,'LineWidth',2), res_slips,num2cell(c,2));
cellfun(@(x) plot(x.time, func(x.slip_ratio), 'LineWidth',2), res_slips);
ylabel('Slip ratio [-]')
legend(folds_suffix);
xlim([0 25]);
grid on;
box;
ylim([-1.18 1.38])
yticks(-1.4:0.2:1.4)

nexttile(2);
hold on;
% cellfun(@(x,cc) plot(x.time, x.slip_angle_deg, 'Color',cc,'LineWidth',2), res_slips,num2cell(c,2));
cellfun(@(x) plot(x.time, func(x.slip_angle_deg), 'LineWidth',2), res_slips);
ylabel('Slip angle [deg]');
xlim([0 25]);
grid on;
box;

%% Traversed distance comparison - review1
% compute traversed distance
res_distance_steps = cellfun(@(x) [vecnorm(diff([x.pos_x,x.pos_y,x.pos_z],1,1),2,2); nan], ...
    res_odom,'UniformOutput',false);

res_distance = cellfun(@(x) cumsum(x), res_distance_steps,'UniformOutput',false);

% plots
figure;
tiledlayout(2,1, "TileSpacing","compact");

% travelled distance of each run
nexttile(1);
hold on;
cellfun(@(x,y) plot(x.time, y, 'LineWidth',2), res_odom,res_distance);
ylabel('Distance [m]');
legend(folds_suffix);
xlim([0 25]);
grid on;
box;

% distance comparison, data of travelled distance  are linearly 
% interpolated in order to compare the same timestamps
refence = find(strcmp(folds_suffix, 'GAZEBO'));
timestamp_compare = (0:0.5:round(min(cellfun(@(x) x.time(end),res_odom))))';

dist_interp = cellfun(@(t,d) interp1(t.time,d,timestamp_compare), ...
    res_odom, res_distance, 'UniformOutput',false);

% % relative distance [%] as plug/gaz * 100
% nexttile(2);
% hold on;
% for ii = 1:length(folds_suffix)
%     if ii == refence
%         continue;
%     end
% 
%     dist_diff = dist_interp{ii} ./ dist_interp{refence} * 100;
%     plot(timestamp_compare, dist_diff, 'LineWidth',2);
% end
% ylabel('Relative distance [%]');
% legend(folds_suffix((1:length(folds_suffix))~=refence));
% xlim([0 25]);
% grid on;
% box;

% Deviation as (plug-gaz)/gaz * 100
% deviation > 0  ->  travelled more than gazebo (downhill)
% deviation < 0  ->  travelled less than gazebo (downhill)
res_deviation = cell(size(dist_interp));
nexttile(2);
hold on;
for ii = 1:length(folds_suffix)
    res_deviation{ii} = (dist_interp{ii} - dist_interp{refence}) ./ dist_interp{refence} * 100;

    if ii == refence
        continue;
    end

    plot(timestamp_compare, res_deviation{ii}, 'LineWidth',2);
end
ylabel('Deviation [%]');
xlabel('Time [s]')
legend(folds_suffix((1:length(folds_suffix))~=refence));
xlim([0 25]);
grid on;
box;

%% Slips statistics - review1

end_time = 25;

% mean and standard deviation
for ii = 1:length(folds_suffix)
    res_slips{ii}.wheel_slip_ratio_mean = nan(1,4);
    res_slips{ii}.wheel_slip_ratio_std = nan(1,4);
    res_slips{ii}.wheel_slip_ang_mean = nan(1,4);
    res_slips{ii}.wheel_slip_ang_std = nan(1,4);

    res_slips{ii}.slip_ratio_concat = [];
    res_slips{ii}.slip_angle_concat = [];

    for jj = 1:4
        indx = res_inter_states{ii,jj}.time <= end_time;

        res_slips{ii}.wheel_slip_ratio_mean(jj) = mean(res_inter_states{ii,jj}.slip_ratio(indx));
        res_slips{ii}.wheel_slip_ratio_std(jj) = std(res_inter_states{ii,jj}.slip_ratio(indx));

        res_slips{ii}.wheel_slip_ang_mean(jj) = mean(res_inter_states{ii,jj}.slip_angle_deg(indx));
        res_slips{ii}.wheel_slip_ang_std(jj) = std(res_inter_states{ii,jj}.slip_angle_deg(indx));

        res_slips{ii}.slip_ratio_concat = [res_slips{ii}.slip_ratio_concat; res_inter_states{ii,jj}.slip_ratio(indx)];
        res_slips{ii}.slip_angle_concat = [res_slips{ii}.slip_angle_concat; res_inter_states{ii,jj}.slip_angle_deg(indx)];
    end
end

% plot
figure;
tiledlayout(2,1, 'TileSpacing','tight');
cc = orderedcolors("gem12");

nexttile(1);
hold on;
cellfun(@(x,i) plot(x.wheel_slip_ratio_mean, '-o', 'Color',cc(i,:), 'LineWidth',2), res_slips,num2cell(1:length(res_slips))');
cellfun(@(x,i) plot(x.wheel_slip_ratio_std, '--*', 'Color',cc(i,:)), res_slips,num2cell(1:length(res_slips))');
ylabel('slip ratio [-]');
legend([strcat("mean ",folds_suffix); strcat("std ",folds_suffix)]);
xlim([0.8 4.2]);
xticks(1:4);
xticklabels(wheels);
grid on;
box;

nexttile(2);
hold on;
cellfun(@(x,i) plot(x.wheel_slip_ang_mean, '-o', 'Color',cc(i,:), 'LineWidth',2), res_slips,num2cell(1:length(res_slips))');
cellfun(@(x,i) plot(x.wheel_slip_ang_std, '--*', 'Color',cc(i,:)), res_slips,num2cell(1:length(res_slips))');
ylabel('slip angle [deg]');
legend([strcat("mean ",folds_suffix); strcat("std ",folds_suffix)]);
xlim([0.8 4.2]);
xticks(1:4);
xticklabels(wheels);
grid on;
box;

