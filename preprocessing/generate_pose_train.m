% This script generates triples from MPII Human Pose dataset
% as well as downscales original images. It also generates
% synthetic distortions of the images to increase model robustness.

function generate_pose_train()
  dataset_path = '../data/mpii_human_pose';

  % Make sure to uncomment only what is necessary (hasn't run yet)

  resize_images(dataset_path, 128, 128);
  generate_triplets(dataset_path);
  exit;
end

function generate_triplets(dataset_path)
 %TODO
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
    in_image_path = strcat(in_images_path, '/', image_name);
    out_image_path = strcat(out_images_path, '/', image_name);

    img = imread(in_image_path);
    img_size = size(img);
    img = im2uint8(img);
    img = imresize(img, [h, w]);
    imwrite(img, out_image_path, 'jpg', 'Quality', 100);

    % update annotation coordinates
    ratio = [h, w]./img_size(1,1:2);
    j=1;
    for j=1:n_annotations
      anno_image_name = RELEASE.annolist(1,j,:).image.name;
      if anno_image_name == image_name
        break;
      end
    end

    if j == n_annotations+1 % not found
      continue;
    end

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
