import os
import time

import numpy as np
import pygame


# Class representing a static grid environment
class GridWorldEnv:
    def __init__(self, seed=None):
        """
        Initialize the static grid environment.

        Args:
            seed (int): Optional random seed for reproducibility.
        """
        self.grid_size = 11  # Size of the grid (11x11)
        self.cell_size = 64  # Pixel size of each cell in the grid (64x64 pixels)

        # Define obstacles as a list of coordinates (x, y)
        # L-shaped pattern in top-left and cross pattern in centre
        self.obstacles = [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2),  # L-shape
                          (5, 4), (5, 5), (5, 6), (4, 5), (6, 5)]  # Cross

        # Define the goal position at the bottom-right corner of the grid
        self.goal = (self.grid_size - 1, self.grid_size - 1)

        self.state = None  # The agent's current position will be set later

        # Set random seed if provided to ensure consistent results
        if seed is not None:
            np.random.seed(seed)

        # Define the action space: 4 possible directions (up, down, left, right)
        self.action_space = 4

        # Observation space is the grid itself, represented by its dimensions
        self.observation_space = (self.grid_size, self.grid_size)

        # Pygame screen initialization is delayed until rendering to avoid immediate window creation
        self.screen = None
        self.render_initialized = False  # Flag to indicate if rendering has been set up

    def reset(self):
        """
        Reset the environment by selecting a random start position for the agent,
        avoiding obstacles and the goal.

        Returns:
            np.array: The initial position of the agent.
        """
        while True:
            # Randomly select starting coordinates (x, y) in the grid
            start_x = np.random.randint(0, self.grid_size)
            start_y = np.random.randint(0, self.grid_size)

            # Ensure the starting position is not on an obstacle or the goal
            if (start_x, start_y) not in self.obstacles and (
                start_x,
                start_y,
            ) != self.goal:
                self.state = (start_x, start_y)
                break

        return self.state

    def step(self, action):
        """
        Take an action in the environment, resulting in a state transition.

        Args:
            action (int): The action taken by the agent (0: up, 1: down, 2: left, 3: right).

        Returns:
            np.array: The new state (position) of the agent.
            int: The reward for the action taken.
            bool: Whether the goal has been reached (episode termination).
            dict: Extra information (unused here).
        """
        x, y = self.state  # Get the agent's current position
        next_x, next_y = x, y  # Initialize the next state as the current state

        # Update position based on the action
        if action == 0 and x > 0:  # Move up
            next_x -= 1
        elif action == 1 and x < self.grid_size - 1:  # Move down
            next_x += 1
        elif action == 2 and y > 0:  # Move left
            next_y -= 1
        elif action == 3 and y < self.grid_size - 1:  # Move right
            next_y += 1

        # Check if the next position is an obstacle
        if (next_x, next_y) in self.obstacles:
            # If the next position is an obstacle, stay in the current position
            next_x, next_y = x, y
            reward = -10  # Penalty for hitting an obstacle
        else:
            reward = -1  # Normal step penalty

        self.state = (next_x, next_y)  # Update the agent's position

        # Check if the agent has reached the goal
        if self.state == self.goal:
            return (
                self.state,
                25,
                True,
                {},
            )  # Reward of 25 for reaching the goal

        return self.state, reward, False, {}  # Continue the episode

    def render(
        self,
        delay=0.1,
        mode="human",
        episode=1,
        learning_type="Q-learning",
        availability=None,
        accuracy=None,
    ):
        """
        Render the grid environment, displaying the agent, goal, and obstacles.
        Also display information such as episode number, learning type, and optionally availability and accuracy.

        Args:
            delay (float): Delay between frames (to control speed of rendering).
            mode (str): Rendering mode (unused in this implementation).
            episode (int): Current episode number.
            learning_type (str): The type of learning algorithm being used (e.g., Q-learning, SARSA).
            availability (float): Teacher availability (optional, as a percentage).
            accuracy (float): Teacher accuracy (optional, as a percentage).
        """
        if not self.render_initialized:
            # Initialize Pygame only when rendering for the first time
            pygame.init()

            # Set the screen size: 50 extra pixels at the top for overlay text
            total_height = self.grid_size * self.cell_size + 50
            self.screen = pygame.display.set_mode(
                (self.grid_size * self.cell_size, total_height)
            )

            # Load images for the agent, goal, and obstacles
            self.agent_img = pygame.image.load(os.path.join("images", "grid_agent.png"))
            self.goal_img = pygame.image.load(os.path.join("images", "grid_goal_position.png"))
            self.obstacle_img = pygame.image.load(
                os.path.join("images", "grid_obstacle.png")
            )

            # Resize images to fit the grid cells
            self.agent_img = pygame.transform.scale(
                self.agent_img, (self.cell_size, self.cell_size)
            )
            self.goal_img = pygame.transform.scale(
                self.goal_img, (self.cell_size, self.cell_size)
            )
            self.obstacle_img = pygame.transform.scale(
                self.obstacle_img, (self.cell_size, self.cell_size)
            )

            # Initialize font for rendering text
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 20)  # Set font style and size

            self.render_initialized = True  # Mark rendering as initialized

        # Handle window close events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Clear the screen with a white background
        self.screen.fill((255, 255, 255))

        # Draw the grid (shift it down by 50 pixels to make space for the text overlay)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * self.cell_size,
                    x * self.cell_size + 50,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(
                    self.screen, (200, 200, 200), rect, 1
                )  # Draw grid lines

        # Draw the goal position
        self.screen.blit(
            self.goal_img,
            (self.goal[1] * self.cell_size, self.goal[0] * self.cell_size + 50),
        )

        # Draw each obstacle
        for obs in self.obstacles:
            self.screen.blit(
                self.obstacle_img,
                (obs[1] * self.cell_size, obs[0] * self.cell_size + 50),
            )

        # Draw the agent's position
        self.screen.blit(
            self.agent_img,
            (self.state[1] * self.cell_size, self.state[0] * self.cell_size + 50),
        )

        # Create a semi-transparent overlay at the top for information text
        overlay_rect = pygame.Surface((self.grid_size * self.cell_size, 50))
        overlay_rect.set_alpha(150)  # Set transparency for the overlay
        overlay_rect.fill((0, 0, 0))  # Black background for the overlay
        self.screen.blit(
            overlay_rect, (0, 0)
        )  # Draw the overlay at the top of the screen

        # Create the title text with episode, learning type, and optionally availability and accuracy
        title = f"Episode: {episode + 1}"
        if availability is not None and accuracy is not None:
            title += f", Availability: {availability * 100:.1f}%, Accuracy: {accuracy * 100:.1f}%"

        # Render the text in white over the black overlay
        text_surface = self.font.render(title, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))  # Position the text at the top left

        # Update the Pygame display
        pygame.display.flip()

        # Introduce a delay to control the speed of rendering
        time.sleep(delay)

    def close(self):
        """
        Close the Pygame window and clean up resources.
        """
        if self.render_initialized:
            pygame.quit()
            self.render_initialized = False
