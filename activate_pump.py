from gpiozero import LED
pump = LED(17)

if output >= 0.5:
    print("✅ 500 coin detected – activating pump")
    pump.on()
    time.sleep(35)
    pump.off()
