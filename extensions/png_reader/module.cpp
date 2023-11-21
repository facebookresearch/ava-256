#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define WUFFS_IMPLEMENTATION
#define WUFFS_CONFIG__STATIC_FUNCTIONS
#define WUFFS_CONFIG__MODULES
#define WUFFS_CONFIG__MODULE__ADLER32
#define WUFFS_CONFIG__MODULE__BASE
#define WUFFS_CONFIG__MODULE__CRC32
#define WUFFS_CONFIG__MODULE__DEFLATE
#define WUFFS_CONFIG__MODULE__PNG
#define WUFFS_CONFIG__MODULE__ZLIB
#include "wuffs.c"


namespace py = pybind11;

typedef py::array_t<uint8_t, py::array::c_style> ndarray_uint8;


PYBIND11_MODULE(png_reader_ext, m)
{
	m.def(
	        "decode_png",
	        +[](const py::buffer& data) {
		        py::buffer_info info = data.request();

		        wuffs_base__io_buffer src = wuffs_base__ptr_u8__reader((uint8_t*) info.ptr, info.size, true);

		        wuffs_png__decoder dec;
		        memset((void*) &dec, 0, sizeof(wuffs_png__decoder));
		        if (wuffs_png__decoder__initialize(&dec, sizeof(wuffs_png__decoder), WUFFS_VERSION, WUFFS_INITIALIZE__ALREADY_ZEROED).repr)
		        {
			        throw std::runtime_error("Failed to initialize wuffs png decoder!");
		        }

		        wuffs_base__image_config ic;
		        wuffs_base__status status = wuffs_png__decoder__decode_image_config(&dec, &ic, &src);
		        if (status.repr)
		        {
			        throw std::runtime_error("Invalid png!");
		        }

		        uint32_t dim_x = wuffs_base__pixel_config__width(&ic.pixcfg);
		        uint32_t dim_y = wuffs_base__pixel_config__height(&ic.pixcfg);
		        size_t num_pixels = dim_x * dim_y;
		        if (num_pixels > (SIZE_MAX / 4))
		        {
			        throw std::runtime_error("Too large png!");
		        }
		        auto pixel_format = wuffs_base__pixel_config__pixel_format(&ic.pixcfg);

		        if (!wuffs_base__pixel_format__is_interleaved(&pixel_format))
		        {
			        throw std::runtime_error("PNG must be interleaved!");
		        }

		        int bpp = wuffs_base__pixel_format__bits_per_pixel(&pixel_format);
		        int channels = 0;
		        uint32_t pixfmt_repr = 0;
		        switch (pixel_format.repr)
		        {
			        case WUFFS_BASE__PIXEL_FORMAT__A:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__A;
				        channels = 1;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__Y:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__Y;
				        channels = 1;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_PREMUL:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_BINARY:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGB;
				        channels = 3;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__BGR:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGB;
				        channels = 3;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL_4X16LE:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__BGRX:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBX;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__RGB:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGB;
				        channels = 3;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL_4X16LE:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL_4X16LE:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY;
				        channels = 4;
				        break;
			        case WUFFS_BASE__PIXEL_FORMAT__RGBX:
				        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBX;
				        channels = 4;
				        break;

			        default: {
				        if (bpp % 8 != 0)
				        {
					        throw std::runtime_error("Wrong bit ber pixel is not divisible by 8 (12 bit PNG?)");
				        }

				        // TODO: figure out more reliable way to get channel count from wuffs
				        channels = bpp / 8;
				        switch (channels)
				        {
					        case 1:
						        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__A; break;
					        case 3:
						        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGB; break;
					        case 4:
						        pixfmt_repr = WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL; break;
					        default:
						        throw std::runtime_error("Number of channels is not supported");
				        }
			        }
		        }

		        wuffs_base__pixel_config__set(
		                &ic.pixcfg,
		                pixfmt_repr,
		                WUFFS_BASE__PIXEL_SUBSAMPLING__NONE,
		                dim_x, dim_y);

		        ndarray_uint8 result;
		        uint8_t* dst_ptr = nullptr;
		        if (channels == 1)
		        {
			        result = ndarray_uint8({(int) dim_y, (int) dim_x});
			        auto r = result.mutable_unchecked<2>();
			        dst_ptr = r.mutable_data(0, 0);
		        }
		        else
		        {
			        result = ndarray_uint8({(int) dim_y, (int) dim_x, (int) channels});
			        auto r = result.mutable_unchecked<3>();
			        dst_ptr = r.mutable_data(0, 0, 0);
		        }

		        size_t workbuf_len = wuffs_png__decoder__workbuf_len(&dec).max_incl;
		        if (workbuf_len > SIZE_MAX)
		        {
			        throw std::runtime_error("Invalid png! workbuf_len > SIZE_MAX");
		        }

		        wuffs_base__slice_u8 workbuf_slice = wuffs_base__make_slice_u8((uint8_t*) malloc(workbuf_len), workbuf_len);
		        if (!workbuf_slice.ptr)
		        {
			        throw std::runtime_error("Falied to allocate workbuf_slice");
		        }

		        wuffs_base__slice_u8 pixbuf_slice = wuffs_base__make_slice_u8(dst_ptr, num_pixels * channels);
		        if (!pixbuf_slice.ptr)
		        {
			        free(workbuf_slice.ptr);
			        throw std::runtime_error("Failed to create pixbuf_slice");
		        }

		        wuffs_base__pixel_buffer pb;
		        status = wuffs_base__pixel_buffer__set_from_slice(&pb, &ic.pixcfg, pixbuf_slice);
		        if (status.repr)
		        {
			        free(workbuf_slice.ptr);
			        throw std::runtime_error("Failed to setup pixel_buffer");
		        }

		        while (true)
		        {
			        wuffs_base__frame_config fc;
			        status = wuffs_png__decoder__decode_frame_config(&dec, &fc, &src);
			        if (status.repr == wuffs_base__note__end_of_data)
			        {
				        break;
			        }
			        status = wuffs_png__decoder__decode_frame(&dec, &pb, &src, WUFFS_BASE__PIXEL_BLEND__SRC, workbuf_slice, nullptr);
			        if (status.repr)
			        {
				        free(workbuf_slice.ptr);
				        throw std::runtime_error("Failed to decode png!");
			        }
		        }

		        free(workbuf_slice.ptr);
		        return result;
	        },
	        "Decode PNG",
	        py::arg("data"));
}
