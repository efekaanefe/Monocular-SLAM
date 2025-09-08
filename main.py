import pygame
import cv2
import numpy as np
from slam import SLAM

WINDOW_WIDTH = 1366
WINDOW_HEIGHT = 768
FPS = 75
VIDEO_FILE = "videos/test1.mp4" 

WIDTH_PERCENT_RIGHT = 0.4
WIDTH_PERCENT_LEFT = 1 - WIDTH_PERCENT_RIGHT

COLOR_BACKGROUND = (30, 30, 40)
COLOR_PANE_BACKGROUND = (10, 10, 15)
COLOR_BORDER = (80, 80, 90)
COLOR_TEXT = (220, 220, 230)

NFEATURES = 2000
DOWNSAMPLE_N_FRAMES = 1
MIN_POINTS_VISUALIZE = 500

class PointCloudVisualizer:
    def __init__(self, pane_rect):
        self.pane_rect = pane_rect
        self.rotation_x = 0
        self.rotation_y = 0
        self.scale = 30
        self.offset_x = pane_rect.width // 2
        self.offset_y = pane_rect.height // 2
        
    def project_3d_to_2d(self, x, y, z):
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.rotation_x), -np.sin(self.rotation_x)],
            [0, np.sin(self.rotation_x), np.cos(self.rotation_x)]
        ])
        Ry = np.array([
            [np.cos(self.rotation_y), 0, np.sin(self.rotation_y)],
            [0, 1, 0],
            [-np.sin(self.rotation_y), 0, np.cos(self.rotation_y)]
        ])
        
        R = Rx @ Ry
        
        point = np.array([x, y, z])
        rotated = R @ point
        
        screen_x = int(rotated[0] * self.scale + self.offset_x)
        screen_y = int(rotated[1] * self.scale + self.offset_y)
        
        return screen_x, screen_y, rotated[2]
        
    
    def draw_point_cloud(self, screen, point_cloud, camera_pose):
        cam_center = np.array([0, 0, 0]) 
        if camera_pose is not None:
            cam_center = camera_pose[:3, 3]
        
        points_3d = [(p[0], p[1], p[2]) for p in point_cloud]
        colors = [p[3] for p in point_cloud]
        
        projected_points = []
        for i, (x, y, z) in enumerate(points_3d):
            rel_x = x - cam_center[0]
            rel_y = y - cam_center[1] 
            rel_z = z - cam_center[2]
            
            screen_x, screen_y, z_depth = self.project_3d_to_2d(rel_x, rel_y, rel_z)
            projected_points.append((screen_x, screen_y, z_depth, colors[i]))
        
        projected_points.sort(key=lambda p: p[2], reverse=True)
        
        for screen_x, screen_y, z_depth, color in projected_points:
            abs_x = self.pane_rect.x + screen_x
            abs_y = self.pane_rect.y + screen_y
            
            if (abs_x < self.pane_rect.x or abs_x > self.pane_rect.x + self.pane_rect.width or
                abs_y < self.pane_rect.y or abs_y > self.pane_rect.y + self.pane_rect.height):
                continue
            
            size = 1
            
            pygame.draw.circle(screen, color, (int(abs_x), int(abs_y)), int(size))

        self.draw_camera_frustum(screen, camera_pose)
        
    def draw_camera_frustum(self, screen, camera_pose, reference_pose=None):
        """
        Draw a camera frustum. If reference_pose is provided, draw relative to that pose.
        Otherwise, draw relative to current view (as before).
        """
        if camera_pose is None:
            return

        frustum_size = 1.6   
        frustum_depth = 0.8

        corners = np.array([
            [-frustum_size/2, -frustum_size/2, frustum_depth],  # bottom-left
            [ frustum_size/2, -frustum_size/2, frustum_depth],  # bottom-right
            [ frustum_size/2,  frustum_size/2, frustum_depth],  # top-right
            [-frustum_size/2,  frustum_size/2, frustum_depth],  # top-left
        ])

        rotation = camera_pose[:3, :3]
        translation = camera_pose[:3, 3]

        world_corners = [(rotation @ corner) + translation for corner in corners]
        world_apex = translation

        # Determine reference point for projection
        if reference_pose is not None:
            cam_center = reference_pose[:3, 3]
        else:
            # Draw relative to the camera's own position 
            cam_center = translation

        # Project apex
        rel_apex = world_apex - cam_center
        apex_screen_x, apex_screen_y, _ = self.project_3d_to_2d(rel_apex[0], rel_apex[1], rel_apex[2])
        apex_screen = (self.pane_rect.x + apex_screen_x, self.pane_rect.y + apex_screen_y)

        # Project corners
        corner_screen_points = []
        for world_corner in world_corners:
            rel_corner = world_corner - cam_center
            corner_x, corner_y, _ = self.project_3d_to_2d(rel_corner[0], rel_corner[1], rel_corner[2])
            corner_screen_points.append((self.pane_rect.x + corner_x, self.pane_rect.y + corner_y))

        frustum_color = (255, 100, 100)  
        line_width = 2
        
        # Draw lines from apex to each corner
        for corner_screen in corner_screen_points:
            pygame.draw.line(screen, frustum_color, apex_screen, corner_screen, line_width)

        # Draw base rectangle
        for i in range(4):
            start = corner_screen_points[i]
            end = corner_screen_points[(i + 1) % 4]
            pygame.draw.line(screen, frustum_color, start, end, line_width)

        # Draw camera center as a filled circle
        pygame.draw.circle(screen, (255, 0, 0), (int(apex_screen[0]), int(apex_screen[1])), 5)

    def draw_multiple_camera_frustums(self, screen, poses, current_pose=None):
        """Draw multiple camera frustums, all relative to the same reference frame"""
        reference_pose = current_pose
            
        for i, pose in enumerate(poses):
            if pose is not None:
                if i == len(poses) - 1 and current_pose is not None and current_pose is pose:
                    self.draw_camera_frustum(screen, pose, reference_pose) # Current pose - draw normally
                else:
                    self._draw_past_camera_frustum(screen, pose, reference_pose) # Past pose - draw with different color

    def _draw_past_camera_frustum(self, screen, camera_pose, reference_pose):
        """Draw a past camera frustum with different styling"""
        if camera_pose is None or reference_pose is None:
            return

        frustum_size = 1.6   
        frustum_depth = 0.8

        corners = np.array([
            [-frustum_size/2, -frustum_size/2, frustum_depth],  # bottom-left
            [ frustum_size/2, -frustum_size/2, frustum_depth],  # bottom-right
            [ frustum_size/2,  frustum_size/2, frustum_depth],  # top-right
            [-frustum_size/2,  frustum_size/2, frustum_depth],  # top-left
        ])

        rotation = camera_pose[:3, :3]
        translation = camera_pose[:3, 3]

        world_corners = [(rotation @ corner) + translation for corner in corners]
        world_apex = translation

        cam_center = reference_pose[:3, 3]  

        rel_apex = world_apex - cam_center
        apex_screen_x, apex_screen_y, _ = self.project_3d_to_2d(rel_apex[0], rel_apex[1], rel_apex[2])
        apex_screen = (self.pane_rect.x + apex_screen_x, self.pane_rect.y + apex_screen_y)

        # Check if apex is out of bounds
        abs_x, abs_y = apex_screen
        if (abs_x < self.pane_rect.x or abs_x > self.pane_rect.x + self.pane_rect.width or
            abs_y < self.pane_rect.y or abs_y > self.pane_rect.y + self.pane_rect.height):
            return 

        corner_screen_points = []
        for world_corner in world_corners:
            rel_corner = world_corner - cam_center
            corner_x, corner_y, _ = self.project_3d_to_2d(rel_corner[0], rel_corner[1], rel_corner[2])
            corner_screen_points.append((self.pane_rect.x + corner_x, self.pane_rect.y + corner_y))

        frustum_color = (100, 100, 255)  # for past poses
        line_width = 1  
        
        for corner_screen in corner_screen_points:
            pygame.draw.line(screen, frustum_color, apex_screen, corner_screen, line_width)

        for i in range(4):
            start = corner_screen_points[i]
            end = corner_screen_points[(i + 1) % 4]
            pygame.draw.line(screen, frustum_color, start, end, line_width)

        pygame.draw.circle(screen, (0, 0, 255), (int(apex_screen[0]), int(apex_screen[1])), 3)

    def handle_input(self, events):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        pane_x, pane_y = self.pane_rect.x, self.pane_rect.y
        mouse_in_pane = (pane_x <= mouse_x < pane_x + self.pane_rect.width and
                         pane_y <= mouse_y < pane_y + self.pane_rect.height)

        keys = pygame.key.get_pressed()
        zoom_speed = 5

        if keys[pygame.K_r]:
            self.rotation_x = 0
            self.rotation_y = 0
            self.scale = 30
            self.offset_x = self.pane_rect.width // 2
            self.offset_y = self.pane_rect.height // 2

        for event in events:
            if event.type == pygame.MOUSEMOTION and mouse_in_pane:
                if pygame.mouse.get_pressed()[0]:  # Rotate with left mouse
                    rel_x, rel_y = event.rel
                    self.rotation_y += rel_x * 0.01
                    self.rotation_x += -rel_y * 0.01  

                elif pygame.mouse.get_pressed()[2]:  # Pan with right mouse
                    rel_x, rel_y = event.rel
                    self.offset_x += rel_x
                    self.offset_y += rel_y

            elif event.type == pygame.MOUSEWHEEL and mouse_in_pane:
                old_scale = self.scale
                self.scale = max(10, min(100, self.scale + event.y * zoom_speed))

                zoom_factor = self.scale / old_scale if old_scale != 0 else 1.0

                mouse_local_x = mouse_x - pane_x
                mouse_local_y = mouse_y - pane_y

                self.offset_x = mouse_local_x - (mouse_local_x - self.offset_x) * zoom_factor
                self.offset_y = mouse_local_y - (mouse_local_y - self.offset_y) * zoom_factor


def main():
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Monocular SLAM Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('-apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif', 24)

    cap = cv2.VideoCapture(VIDEO_FILE)

    point_cloud_pane_rect = pygame.Rect(
        0,
        0,
        int(WINDOW_WIDTH * WIDTH_PERCENT_LEFT),
        WINDOW_HEIGHT
    )

    video_pane_rect = pygame.Rect(
        int(WINDOW_WIDTH * WIDTH_PERCENT_LEFT),
        0,
        int(WINDOW_WIDTH * WIDTH_PERCENT_RIGHT),
        WINDOW_HEIGHT // 2
    )

    features_pane_rect = pygame.Rect(
        int(WINDOW_WIDTH * WIDTH_PERCENT_LEFT),
        WINDOW_HEIGHT // 2,
        int(WINDOW_WIDTH * WIDTH_PERCENT_RIGHT),
        WINDOW_HEIGHT // 2
    )

    point_cloud_viz = PointCloudVisualizer(point_cloud_pane_rect)

    # camera_matrix = np.array([
    #     [500, 0, 320],  # fx, 0, cx
    #     [0, 500, 240],  # 0, fy, cy  
    #     [0, 0, 1]       # 0, 0, 1
    # ])

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fov_degrees = 160
    fov_rad = np.radians(fov_degrees)
    fx = (width / 2) / np.tan(fov_rad / 2)
    fy = fx
    cx = width / 2
    cy = height / 2
    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)


    my_slam = SLAM(camera_matrix=camera_matrix, nfeatures=NFEATURES)

    running = True
    frame_count = 0
    
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        point_cloud_viz.handle_input(events)

        ret, frame = cap.read()
        if not ret:
            print("Video ended. Looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()


        frame_count += 1
        if frame_count % DOWNSAMPLE_N_FRAMES != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))  
        frame_rgb = np.flipud(frame_rgb)  

        screen.fill(COLOR_BACKGROUND)

        def draw_pane(rect, title):
            pygame.draw.rect(screen, COLOR_PANE_BACKGROUND, rect)
            pygame.draw.rect(screen, COLOR_BORDER, rect, 2)
            text_surface = font.render(title, True, COLOR_TEXT)
            screen.blit(text_surface, (rect.x + 10, rect.y + 10))

        draw_pane(point_cloud_pane_rect, "3D Point Cloud")
        draw_pane(video_pane_rect, "Original Video Feed")
        draw_pane(features_pane_rect, "Video with Features")

        # --- Display Video Content ---
        video_surface = pygame.surfarray.make_surface(frame_rgb)
        scaled_video_surface = pygame.transform.scale(video_surface, video_pane_rect.size)
        screen.blit(scaled_video_surface, video_pane_rect.topleft)

        curr_pose, keypoints = my_slam.process_frame(frame_rgb, min_points=MIN_POINTS_VISUALIZE)
        frame_with_keypoints = my_slam.get_frame_with_keypoints(frame_rgb, keypoints)

        features_surface = pygame.surfarray.make_surface(frame_with_keypoints)
        scaled_features_surface = pygame.transform.scale(features_surface, features_pane_rect.size)
        screen.blit(scaled_features_surface, features_pane_rect.topleft)

        # --- Draw Point Cloud ---
        point_cloud = my_slam.get_point_cloud()
        point_cloud_viz.draw_point_cloud(screen, point_cloud, curr_pose)

        # --- Draw all camera poses ---
        poses = my_slam.get_poses()
        point_cloud_viz.draw_multiple_camera_frustums(screen, poses, curr_pose)

        # # --- Draw trajectory info ---
        # trajectory = my_slam.get_trajectory()
        # if len(trajectory) > 0:
        #     pos_text = font.render(
        #         f"Position: ({trajectory[-1][0]:.2f}, {trajectory[-1][1]:.2f}, {trajectory[-1][2]:.2f})", 
        #         True, COLOR_TEXT
        #     )
        #     screen.blit(pos_text, (point_cloud_pane_rect.x + 10, point_cloud_pane_rect.y + 40))

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
