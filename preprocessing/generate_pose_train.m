% This script generates triples from MPII Human Pose dataset
% as well as downscales original images. It also generates
% synthetic distortions of the images to increase model robustness.

% TODO: split train and test sets
function generate_pose_train()
  dataset_path = '../data/mpii_human_pose';

  % Make sure to uncomment only what is necessary (hasn't run yet)

  resize_images(dataset_path, 128, 128);
  generate_triplets(dataset_path, 7, 15);
end

function generate_triplets(dataset_path, pos_thresh, neg_thresh)
  annotations_path = strcat(dataset_path, '/annotations.mat');
  triplets_file = fopen(strcat(dataset_path, '/triplets.csv'), 'w');

  load(annotations_path);

  % calculate distance matrix
  n_annotations = size(RELEASE.annolist,2);
  dist = zeros(n_annotations, n_annotations);

  fprintf('Calculating distance matrix\n');
  for i=1:n_annotations
    for j=i+1:n_annotations
      dist(i,j) = pose_distance(RELEASE, i, j);
    end
    if mod(i,50) == 0
      fprintf('Processed %d%% (%d/%d)\n', int32(100*i/n_annotations), ...
              i, n_annotations);
    end
  end
  dist = dist + dist';
  dist = dist + (eye(n_annotations) .* -1);
  % generate triplets
  fprintf('Generating triplets\n');
  dist = int32(dist);
  cnt = 0;
  for i=1:n_annotations
    positives = find(dist(i,:) < pos_thresh & dist(i,:) ~= -1);
    negatives = find(dist(i,:) > neg_thresh);
    for p=1:size(positives,2)
      for n=1:size(negatives,2)
        anchor_name = RELEASE.annolist(1,i,:).image.name;
        pidx = positives(1,p);
        nidx = negatives(1,n);
        pos_name = RELEASE.annolist(1,pidx,:).image.name;
        neg_name = RELEASE.annolist(1,nidx,:).image.name;
        fprintf(triplets_file, '%s %s %s %d %d\n', ...
                anchor_name, pos_name, neg_name, dist(i,pidx), dist(i,nidx));
        cnt = cnt + 1;
      end
    end
    if mod(i,50) == 0
      fprintf('Processed %d%% (%d/%d)\n', int32(100*i/n_annotations), ...
              i, n_annotations);
    end
  end
  fprintf('%d triplets generated\n', cnt);
  fclose(triplets_file);
end

% returns 0 if not found
function idx = find_anno_img_idx(annotations, image_name)
  n_annotations = size(annotations.annolist,2);
  idx = 1;
  for idx=1:n_annotations
    anno_image_name = annotations.annolist(1,idx,:).image.name;
    if anno_image_name == image_name
      break;
    end
  end

  if idx == n_annotations+1 % not found
    idx = 0;
  end
end

function d = pose_distance(annotations, idx1, idx2)
  % TODO: scale both to a common value
  %       in order to make pose_distance robust
  % TODO: multiple people?
  d = -1;
  try
    joints1 = annotations.annolist(1,idx1,:).annorect(1).annopoints.point;
    joints2 = annotations.annolist(1,idx2,:).annorect(1).annopoints.point;
    root = 7; % thorax
    root_idx1 = find_joint(joints1, root);
    root_idx2 = find_joint(joints2, root);
    dx = joints1(1,root_idx1).x - joints2(1,root_idx2).x;
    dy = joints1(1,root_idx1).y - joints2(1,root_idx2).y;

    scale1 = annotations.annolist(1,idx1,:).annorect(1).scale;
    scale2 = annotations.annolist(1,idx2,:).annorect(1).scale;
    ds = scale1 / scale2;

    avg_diff = 0;
    cnt = 0;
    for i=1:size(joints1,2)
      try
        j = find_joint(joints2, joints1(1,i).id);
        x1 = joints1(1,i).x;
        y1 = joints1(1,i).y;
        x2 = (ds * (joints2(1,j).x + dx));
        y2 = (ds * (joints2(1,j).y + dy));
        dxx = abs(x1 - x2);
        dyy = abs(y1 - y2);
        diff = sqrt(dxx^2 + dyy^2);
        avg_diff = avg_diff + diff;
        cnt = cnt+1;
      catch ME
        disp(ME.message); % debug
      end
    end
    if cnt ~= 0
      avg_diff = avg_diff / cnt;
    end
    d = avg_diff;
  catch ME
    disp(ME.message); % debug
  end
end

% id - joint type
function idx = find_joint(joints, id)
  for i=1:size(joints,2)
    if joints(1,i).id == id
      idx = i;
      return;
    end
  end
  error('root not found');
end

function resize_images(dataset_path, w, h)
  in_images_path = strcat(dataset_path, '/orig/images');
  out_images_path = strcat(dataset_path, '/images');

  in_annotations_path = strcat(dataset_path, '/orig/annotations.mat');
  out_annotations_path = strcat(dataset_path, '/annotations.mat');

  load(in_annotations_path);
  n_annotations = size(RELEASE.annolist,2);

  if ~isdir(out_images_path)
    mkdir(out_images_path);
  end

  images = dir(in_images_path);

  n_images = size(images,1);
  for i=3:n_images
    image_name = images(i,:).name;

    j = find_anno_img_idx(RELEASE, image_name);
    if j == 0
      continue;
    end

    in_image_path = strcat(in_images_path, '/', image_name);
    out_image_path = strcat(out_images_path, '/', image_name);

    img = imread(in_image_path);
    img_size = size(img);
    img = im2uint8(img);
    img = imresize(img, [h, w]);
    imwrite(img, out_image_path, 'jpg', 'Quality', 100);

    % update annotation coordinates
    ratio = [h, w]./img_size(1,1:2);

    anno = RELEASE.annolist(1,j,:);
    n_people = size(anno.annorect,2);

    for k=1:n_people
      body = anno.annorect(1,k,:);
      % warning: ugly code ahead
      % TODO: learn to use references in matlab

      % TODO: we possibly care only about joints

      try
        % head rectangle
        RELEASE.annolist(1,j,:).annorect(1,k,:).x1 = ratio(1,2) * body.x1;
        RELEASE.annolist(1,j,:).annorect(1,k,:).x2 = ratio(1,2) * body.x2;
        RELEASE.annolist(1,j,:).annorect(1,k,:).y1 = ratio(1,1) * body.y1;
        RELEASE.annolist(1,j,:).annorect(1,k,:).y2 = ratio(1,1) * body.y2;

        % TODO: person scale

        % human position
        RELEASE.annolist(1,j,:).annorect(1,k,:).objpos.x = ratio(1,2) * body.objpos.x;
        RELEASE.annolist(1,j,:).annorect(1,k,:).objpos.y = ratio(1,1) * body.objpos.y;

        % joints
        n_joints = size(body.annopoints.point,2);
        for l=1:n_joints
          joint = body.annopoints.point(1,l,:);
          RELEASE.annolist(1,j,:).annorect(1,k,:).annopoints.point(1,l,:).x = ratio(1,2) * joint.x;
          RELEASE.annolist(1,j,:).annorect(1,k,:).annopoints.point(1,l,:).y = ratio(1,1) * joint.y;
        end
      catch % skip invalid annotation structures
        continue;
      end
    end

    if mod(i,50) == 0
      fprintf('Processed %d%% (%d/%d)\n', int32(100*i/n_images), i, n_images);
    end
  end
  save(out_annotations_path, 'RELEASE');
end
