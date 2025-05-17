/*
Compile with something like:
    gcc -shared -fPIC raspberrypi_capture.c -o libraspberrypi_capture.so
Make sure to link any libraries (e.g., -lpthread) if needed, depending on your environment.
*/

#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>
#include <limits.h>
#include <errno.h>
#include <string.h>
#include <time.h>

#define VOSPI_FRAME_SIZE 164
#define LEPTON_FRAME_COLS 80
#define LEPTON_FRAME_ROWS 60

// SPI settings
static const char *device = "/dev/spidev0.1";
static uint8_t mode = 0;
static uint8_t bits = 8;
static uint32_t speed = 16000000;
static uint16_t delay_us = 0;

static void pabort(const char *s)
{
    perror(s);
    // You could return an error instead of aborting to let the caller handle it.
    abort();
}

/**
 * transfer() reads one Lepton packet (164 bytes) into lepton_frame_packet.
 * If it's a valid packet (first nibble != 0x0F), it stores the 80 16-bit pixels
 * into lepton_image[row][col].
 */
static int transfer(int fd,
                    uint8_t *lepton_frame_packet,
                    uint16_t lepton_image[LEPTON_FRAME_ROWS][LEPTON_FRAME_COLS])
{
    uint8_t tx[VOSPI_FRAME_SIZE] = {0};
    struct spi_ioc_transfer tr = {
        .tx_buf = (unsigned long)tx,
        .rx_buf = (unsigned long)lepton_frame_packet,
        .len = VOSPI_FRAME_SIZE,
        .delay_usecs = delay_us,
        .speed_hz = speed,
        .bits_per_word = bits,
    };

    int ret = ioctl(fd, SPI_IOC_MESSAGE(1), &tr);
    if (ret < 1)
        pabort("can't send spi message");

    // Packet header:
    //  [0] = packet number MSB & discard nibble
    //  [1] = packet number LSB
    //  If [0] & 0x0f == 0x0f => it's a discard packet
    if (((lepton_frame_packet[0] & 0x0F) != 0x0F)) {
        int frame_number = lepton_frame_packet[1];
        if (frame_number < LEPTON_FRAME_ROWS) {
            // Copy 80 16-bit pixels
            for (int i = 0; i < LEPTON_FRAME_COLS; i++) {
                lepton_image[frame_number][i] =
                    (lepton_frame_packet[2*i + 4] << 8) |
                     lepton_frame_packet[2*i + 5];
            }
        }
        return frame_number;
    }
    // If invalid packet, return something that won't be recognized as the last row
    return -1;
}

/**
 * main() is the function Python will call via ctypes:
 *
 *  int main(uint8_t *buffer)
 * 
 *  - buffer points to a 9600-byte array in Python:
 *    60 rows * 80 cols * 2 bytes per pixel = 9600
 *  - We capture a single 60x80 frame from the Lepton and copy it to buffer.
 *  - Return 0 on success, or nonzero on error.
 */
int main(uint8_t *buffer)
{
    int fd = open(device, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Error: cannot open device %s: %s\n", device, strerror(errno));
        return -1;
    }

    // Configure SPI settings (mode, bits, speed, etc.)
    if (ioctl(fd, SPI_IOC_WR_MODE, &mode) == -1) {
        pabort("can't set spi mode");
    }
    if (ioctl(fd, SPI_IOC_RD_MODE, &mode) == -1) {
        pabort("can't get spi mode");
    }
    if (ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits) == -1) {
        pabort("can't set bits per word");
    }
    if (ioctl(fd, SPI_IOC_RD_BITS_PER_WORD, &bits) == -1) {
        pabort("can't get bits per word");
    }
    if (ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) == -1) {
        pabort("can't set max speed hz");
    }
    if (ioctl(fd, SPI_IOC_RD_MAX_SPEED_HZ, &speed) == -1) {
        pabort("can't get max speed hz");
    }

    // Allocate and initialize local arrays
    uint8_t  lepton_frame_packet[VOSPI_FRAME_SIZE];
    uint16_t lepton_image[LEPTON_FRAME_ROWS][LEPTON_FRAME_COLS] = {{0}};
    
    int frame_number = -1;
    time_t start_time = time(NULL);  // Start timeout timer
    #define TIMEOUT_SECONDS 0.5  // Set desired timeout duration

    // Replace the existing loop with this one that includes a timeout
    while (frame_number != (LEPTON_FRAME_ROWS - 1)) {
        if (difftime(time(NULL), start_time) > TIMEOUT_SECONDS) {
            fprintf(stderr, "Timeout waiting for final row\n");
            close(fd);
            return -1;  // or handle the timeout error as needed
        }
        frame_number = transfer(fd, lepton_frame_packet, lepton_image);
        // If packet is a discard packet, continue immediately.
        if (frame_number < 0) {
            continue;
        }
    }

    // Copy the 16-bit pixels into the provided buffer (with proper little-endian order)
    for (int row = 0; row < LEPTON_FRAME_ROWS; row++) {
        for (int col = 0; col < LEPTON_FRAME_COLS; col++) {
            uint16_t val = lepton_image[row][col];
            int idx = (row * LEPTON_FRAME_COLS + col) * 2;
            buffer[idx + 0] = (uint8_t)(val & 0xFF);         // LSB
            buffer[idx + 1] = (uint8_t)((val >> 8) & 0xFF);    // MSB
        }
    }

    close(fd);
    return 0;  // success
}
