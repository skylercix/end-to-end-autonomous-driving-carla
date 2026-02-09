import carla

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    print("Connected to:", world.get_map().name)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("model3")[0]

    spawn_points = world.get_map().get_spawn_points()

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    if vehicle is None:
        print("Failed to spawn the vehicle!")
        return

    print("Vehicle spawned!")
    vehicle.set_autopilot(True)

    print("Autopilot enabled. Letting it drive for 10 seconds...")
    import time
    time.sleep(10)

    print("Destroying vehicle...")
    vehicle.destroy()
    print("Done!")

if __name__ == '__main__':
    main()
