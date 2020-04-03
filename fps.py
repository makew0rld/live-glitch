from time import time

class FPS:
    """Provides the FPS of a camera or other device."""

    def __init__(self, avg_period=1):
        """Init the class.
        
        avg_period: The time period before the FPS averaging
                    restarts, in seconds.
        """
        
        self.avg_period = avg_period
        self.start_time = 0
        self.running = False
        self.avg_fpp = 0  # Average frames per period
        self.fpp = 0  # The FPS over avg_period time
    
    def start(self):
        self.start_time = time()
        self.running = True
        return self
    
    def stop(self):
        self.running = False
    
    def update(self):
        temp_time = time()
        if self.running:
            if temp_time - self.start_time > self.avg_period:
                # Enough time has passed to restart the averaging
                self.start_time = temp_time
                # Calculate the new average, using weights
                self.avg_fpp = 0 * self.avg_fpp + 1 * self.fpp
                self.fpp = 0

            self.fpp += 1

    def fps(self):
        return self.avg_fpp / self.avg_period
