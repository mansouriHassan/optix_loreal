
/* ---------------------------------------------------------
 * Hair file 
 * ---------------------------------------------------------
 * Based on Cem Yuksel samples
 * ---------------------------------------------------------
 */

#ifndef _CY_HAIR_FILE_H_INCLUDED_
#define _CY_HAIR_FILE_H_INCLUDED_

/* ------------------------ Libraries ---------------------- */

#include <stdio.h>
#include <math.h>
#include <string.h>

#include <iostream>

/* ----------------------- Definitions --------------------- */

namespace cy {

	#define _CY_HAIR_FILE_SEGMENTS_BIT		1
	#define _CY_HAIR_FILE_POINTS_BIT		2
	#define _CY_HAIR_FILE_THICKNESS_BIT		4
	#define _CY_HAIR_FILE_TRANSPARENCY_BIT	8
	#define _CY_HAIR_FILE_COLORS_BIT		16

	#define _CY_HAIR_FILE_INFO_SIZE			88

	// File read errors
	#define CY_HAIR_FILE_ERROR_CANT_OPEN_FILE		-1	// Cannot open file
	#define CY_HAIR_FILE_ERROR_CANT_READ_HEADER		-2	// Cannot read header
	#define	CY_HAIR_FILE_ERROR_WRONG_SIGNATURE		-3	// Wrong signature
	#define	CY_HAIR_FILE_ERROR_READING_SEGMENTS		-4	// Failed reading segments
	#define	CY_HAIR_FILE_ERROR_READING_POINTS		-5	// Failed reading points
	#define	CY_HAIR_FILE_ERROR_READING_THICKNESS	-6	// Failed reading thickness
	#define	CY_HAIR_FILE_ERROR_READING_TRANSPARENCY	-7	// Failed reading transparency
	#define	CY_HAIR_FILE_ERROR_READING_COLORS		-8	// Failed reading colors
	 
/* ------------------------ Hair File ---------------------- */

	//Hair file class
	class HairFile {

	public:

		/* Constructors ------------------------------------ */
		HairFile() : segments(nullptr), points(nullptr), thickness(nullptr),
			transparency(nullptr), colors(nullptr) { initialize(); }
		~HairFile() { initialize(); }

		/* Hair file header -------------------------------- */
		struct Header {
			// Should be "HAIR"
			char			signature[4];
			// Number of hair strands
			unsigned int	hair_count;
			// Total number of points of all strands
			unsigned int	point_count;
			// Bit array of data in the file
			unsigned int	arrays;
			
			// Default number of segments of each strand
			unsigned int	d_segments;
			// Default thickness of hair strands
			float			d_thickness;
			// Default transparency of hair strands
			float			d_transparency;
			// Default color of hair strands
			float			d_color[3];

			// Information about the file
			char			info[_CY_HAIR_FILE_INFO_SIZE];
		};


		/* Constant data access methods -------------------- */
		// Access header data
		const Header& getHeader() const { return header; }
		// Returns segments array (segment count for each hair strand)
		const unsigned short* getSegmentsArray() const { return segments; }
		// Returns points array (xyz coordinates of each hair point)
		const float* getPointsArray() const { return points; }
		// Returns thickness array (thickness at each hair point}
		const float* getThicknessArray() const { return thickness; }
		// Returns transparency array (transparency at each hair point)
		const float* getTransparencyArray() const { return transparency; }
		// Returns colors array (rgb color at each hair point)
		const float* getColorsArray() const { return colors; }


		/* Data access methods ----------------------------- */
		// Returns segments array (segment count for each hair strand)
		unsigned short* getSegmentsArray() { return segments; }
		// Returns points array (xyz coordinates of each hair point)
		float* getPointsArray() { return points; }
		// Returns thickness array (thickness at each hair point}
		float* getThicknessArray() { return thickness; }
		// Returns transparency array (transparency at each hair point)
		float* getTransparencyArray() { return transparency; }
		// Returns colors array (rgb color at each hair point)
		float* getColorsArray() { return colors; }

		/* Methods for setting array sizes ----------------- */

		// Deletes all arrays and initializes the header data
		void initialize() {
			if (segments) delete[] segments;
			if (points) delete[] points;
			if (colors) delete[] colors;
			if (thickness) delete[] thickness;
			if (transparency) delete[] transparency;
			header.signature[0] = 'h';
			header.signature[1] = 'a';
			header.signature[2] = 'i';
			header.signature[3] = 'r';
			header.hair_count = 0;
			header.point_count = 0;
			header.arrays = 0;	// No arrays
			header.d_segments = 0;
			header.d_thickness = 1.0f;
			header.d_transparency = 0.0f;
			header.d_color[0] = 1.0f;
			header.d_color[1] = 1.0f;
			header.d_color[2] = 1.0f;
			memset(header.info, '\0', _CY_HAIR_FILE_INFO_SIZE);
		}

		// Sets the hair count, re-allocates segments array if necessary
		void setHairCount(int count) {
			header.hair_count = count;
			if (segments) {
				delete[] segments;
				segments = new unsigned short[header.hair_count];
			}
		}

		// Sets the point count, re-allocates points, thickness,
		// transparency, and colors arrays if necessary
		void setPointCount(int count){
			header.point_count = count;
			if (points) {
				delete[] points;
				points = new float[header.point_count * 3];
			}
			if (thickness) {
				delete[] thickness;
				thickness = new float[header.point_count];
			}
			if (transparency) {
				delete[] transparency;
				transparency = new float[header.point_count];
			}
			if (colors) {
				delete[] colors;
				colors = new float[header.point_count * 3];
			}
		}

		// Use this function to allocate/delete arrays, before you call
		// this method set hair count and point count
		// Note that a valid HAIR file should always have points array
		void setArrays(int array_types) {
			header.arrays = array_types;
			if ((header.arrays & _CY_HAIR_FILE_SEGMENTS_BIT) && !segments) {
				segments = new unsigned short[header.hair_count]; 
			}
			if (!(header.arrays & _CY_HAIR_FILE_SEGMENTS_BIT) && segments) {
				delete[] segments; segments = nullptr; 
			}
			if ((header.arrays & _CY_HAIR_FILE_POINTS_BIT) && !points) {
				points = new float[header.point_count * 3]; 
			}
			if (!(header.arrays & _CY_HAIR_FILE_POINTS_BIT) && points) {
				delete[] points; points = nullptr;
			}
			if ((header.arrays & _CY_HAIR_FILE_THICKNESS_BIT) && !thickness) {
				thickness = new float[header.point_count]; 
			}
			if (!(header.arrays & _CY_HAIR_FILE_THICKNESS_BIT) && thickness) {
				delete[] thickness; thickness = nullptr; 
			}
			if ((header.arrays & _CY_HAIR_FILE_TRANSPARENCY_BIT) && !transparency) {
				transparency = new float[header.point_count]; 
			}
			if (!(header.arrays & _CY_HAIR_FILE_TRANSPARENCY_BIT) && transparency) {
				delete[] transparency; transparency = nullptr; 
			}
			if ((header.arrays & _CY_HAIR_FILE_COLORS_BIT) && !colors) {
				colors = new float[header.point_count * 3]; 
			}
			if (!(header.arrays & _CY_HAIR_FILE_COLORS_BIT) && colors) {
				delete[] colors; colors = nullptr; 
			}
		}

		// Sets default number of segments for all hair strands,
		// which is used if segments array does not exist
		void setDefaultSegmentCount(int s) { header.d_segments = s; }

		// Sets default hair strand thickness, which is used if
		// thickness array does not exist
		void setDefaultThickness(float t) { header.d_thickness = t; }

		// Sets default hair strand transparency, which is used
		// if transparency array does not exist
		void setDefaultTransparency(float t) { header.d_transparency = t; }

		// Sets default hair color, which is used if color array
		// does not exist
		void setDefaultColor(float r, float g, float b) {
			header.d_color[0] = r;
			header.d_color[1] = g;
			header.d_color[2] = b; 
		}

		/* Load and save methods --------------------------- */
		
		// Loads hair data from the given HAIR file
		int loadFromFile(const char *filename) {
			initialize();

			FILE *fp;
			fp = fopen(filename, "rb");
			if (fp == nullptr) return CY_HAIR_FILE_ERROR_CANT_OPEN_FILE;

			// Read the header
			size_t headread = fread(&header, sizeof(Header), 1, fp);

			#define _CY_FAILED_RETURN(errorno) { initialize(); fclose( fp ); return errorno; }

			// Check if header is correctly read
			if (headread < 1) _CY_FAILED_RETURN(CY_HAIR_FILE_ERROR_CANT_READ_HEADER);

			// Check if this is a hair file
			if (strncmp(header.signature, "hair", 4) != 0 &&
				strncmp(header.signature, "HAIR", 4) != 0)
				_CY_FAILED_RETURN(CY_HAIR_FILE_ERROR_WRONG_SIGNATURE);

			// Read segments array
			if (header.arrays & _CY_HAIR_FILE_SEGMENTS_BIT) {
				segments = new unsigned short[header.hair_count];
				size_t readcount = fread(segments, sizeof(unsigned short), 
					header.hair_count, fp);
				if (readcount < header.hair_count)
					_CY_FAILED_RETURN(CY_HAIR_FILE_ERROR_READING_SEGMENTS);
			}

			// Read points array
			if (header.arrays & _CY_HAIR_FILE_POINTS_BIT) {
				points = new float[header.point_count * 3];
				size_t readcount = fread(points, sizeof(float),
					header.point_count * 3, fp);
				if (readcount < header.point_count * 3)
					_CY_FAILED_RETURN(CY_HAIR_FILE_ERROR_READING_POINTS);
			}

			// Read thickness array
			if (header.arrays & _CY_HAIR_FILE_THICKNESS_BIT) {
				thickness = new float[header.point_count];
				size_t readcount = fread(thickness, sizeof(float),
					header.point_count, fp);
				if (readcount < header.point_count)
					_CY_FAILED_RETURN(CY_HAIR_FILE_ERROR_READING_THICKNESS);
			}

			// Read thickness array
			if (header.arrays & _CY_HAIR_FILE_TRANSPARENCY_BIT) {
				transparency = new float[header.point_count];
				size_t readcount = fread(transparency, sizeof(float),
					header.point_count, fp);
				if (readcount < header.point_count)
					_CY_FAILED_RETURN(CY_HAIR_FILE_ERROR_READING_TRANSPARENCY);
			}

			// Read colors array
			if (header.arrays & _CY_HAIR_FILE_COLORS_BIT) {
				colors = new float[header.point_count * 3];
				size_t readcount = fread(colors, sizeof(float),
					header.point_count * 3, fp);
				if (readcount < header.point_count * 3) 
					_CY_FAILED_RETURN(CY_HAIR_FILE_ERROR_READING_COLORS);
			}

			fclose(fp);

			return header.hair_count;
		}

		// Saves hair data to the given HAIR file
		int saveToFile(const char *filename) const {
			FILE *fp;
			fp = fopen(filename, "wb");
			if (fp == nullptr) return -1;

			// Write header
			fwrite(&header, sizeof(Header), 1, fp);

			// Write arrays
			if (header.arrays & _CY_HAIR_FILE_SEGMENTS_BIT) 
				fwrite(segments, sizeof(unsigned short), header.hair_count, fp);
			if (header.arrays & _CY_HAIR_FILE_POINTS_BIT) 
				fwrite(points, sizeof(float), header.point_count * 3, fp);
			if (header.arrays & _CY_HAIR_FILE_THICKNESS_BIT) 
				fwrite(thickness, sizeof(float), header.point_count, fp);
			if (header.arrays & _CY_HAIR_FILE_TRANSPARENCY_BIT) 
				fwrite(transparency, sizeof(float), header.point_count, fp);
			if (header.arrays & _CY_HAIR_FILE_COLORS_BIT) 
				fwrite(colors, sizeof(float), header.point_count * 3, fp);

			fclose(fp);

			return header.hair_count;
		}

		/* Other methods ----------------------------------- */

		// Fills the given direction array with normalized directions using
		// the points array, call this function if you need strand directions
		// for shading (the given array dir should be allocated as an array
		// of size 3 times point count). Returns point count, returns zero if fails
		int fillDirectionArray(float *dir) {
			if (dir == nullptr || header.point_count <= 0 || points == nullptr)
				return 0;

			// Point index
			int p = 0;
			for (unsigned int i = 0; i<header.hair_count; i++) {
				int s = (segments) ? segments[i] : header.d_segments;
				if (s > 1) {
					// Direction at point 1
					float len0, len1;
					computeDirection(&dir[(p + 1) * 3], len0, len1, &points[p * 3],
						&points[(p + 1) * 3], &points[(p + 2) * 3]);

					// Direction at point 0
					float d0[3];
					d0[0] = points[(p + 1) * 3] - dir[(p + 1) * 3] *
						len0*0.3333f - points[p * 3];
					d0[1] = points[(p + 1) * 3 + 1] - dir[(p + 1) * 3 + 1] *
						len0*0.3333f - points[p * 3 + 1];
					d0[2] = points[(p + 1) * 3 + 2] - dir[(p + 1) * 3 + 2] *
						len0*0.3333f - points[p * 3 + 2];
					float d0lensq = d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2];
					float d0len = (d0lensq > 0) ? (float)sqrt(d0lensq) : 1.0f;
					dir[p * 3] = d0[0] / d0len;
					dir[p * 3 + 1] = d0[1] / d0len;
					dir[p * 3 + 2] = d0[2] / d0len;

					// Computed the first 2 points
					p += 2;

					// Compute the direction for the rest
					for (int t = 2; t<s; t++, p++) {
						computeDirection(&dir[p * 3], len0, len1, &points[(p - 1) * 3],
							&points[p * 3], &points[(p + 1) * 3]);
					}

					// Direction at the last point
					d0[0] = -points[(p - 1) * 3] + dir[(p - 1) * 3] *
						len1*0.3333f + points[p * 3];
					d0[1] = -points[(p - 1) * 3 + 1] + dir[(p - 1) * 3 + 1] *
						len1*0.3333f + points[p * 3 + 1];
					d0[2] = -points[(p - 1) * 3 + 2] + dir[(p - 1) * 3 + 2] *
						len1*0.3333f + points[p * 3 + 2];
					d0lensq = d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2];
					d0len = (d0lensq > 0) ? (float)sqrt(d0lensq) : 1.0f;
					dir[p * 3] = d0[0] / d0len;
					dir[p * 3 + 1] = d0[1] / d0len;
					dir[p * 3 + 2] = d0[2] / d0len;
					p++;
				}

				// If it has a single segment
				else if (s > 0) {
					float d0[3];
					d0[0] = points[(p + 1) * 3] - points[p * 3];
					d0[1] = points[(p + 1) * 3 + 1] - points[p * 3 + 1];
					d0[2] = points[(p + 1) * 3 + 2] - points[p * 3 + 2];
					float d0lensq = d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2];
					float d0len = (d0lensq > 0) ? (float)sqrt(d0lensq) : 1.0f;
					dir[p * 3] = d0[0] / d0len;
					dir[p * 3 + 1] = d0[1] / d0len;
					dir[p * 3 + 2] = d0[2] / d0len;
					dir[(p + 1) * 3] = dir[p * 3];
					dir[(p + 1) * 3 + 1] = dir[p * 3 + 1];
					dir[(p + 1) * 3 + 2] = dir[p * 3 + 2];
					p += 2;
				}
			}
			return p;
		}

	private:

		/* Variables -------------------------------------- */

		Header header;
		unsigned short *segments;
		float *points;
		float *thickness;
		float *transparency;
		float *colors;

		/* Other private methods --------------------------- */

		// Given point before (p0) and after (p2), computes the direction (d) at p1
		float computeDirection(float *d, float &d0len, float &d1len,
			const float *p0, const float *p1, const float *p2) {
			// Line from p0 to p1
			float d0[3];
			d0[0] = p1[0] - p0[0];
			d0[1] = p1[1] - p0[1];
			d0[2] = p1[2] - p0[2];
			float d0lensq = d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2];
			d0len = (d0lensq > 0) ? (float)sqrt(d0lensq) : 1.0f;

			// Line from p1 to p2
			float d1[3];
			d1[0] = p2[0] - p1[0];
			d1[1] = p2[1] - p1[1];
			d1[2] = p2[2] - p1[2];
			float d1lensq = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
			d1len = (d1lensq > 0) ? (float)sqrt(d1lensq) : 1.0f;

			// Make sure that d0 and d1 has the same length
			d0[0] *= d1len / d0len;
			d0[1] *= d1len / d0len;
			d0[2] *= d1len / d0len;

			// Direction at p1
			d[0] = d0[0] + d1[0];
			d[1] = d0[1] + d1[1];
			d[2] = d0[2] + d1[2];
			float dlensq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
			float dlen = (dlensq > 0) ? (float)sqrt(dlensq) : 1.0f;
			d[0] /= dlen;
			d[1] /= dlen;
			d[2] /= dlen;

			return d0len;
		}
	};
}

typedef cy::HairFile cyHairFile;

#endif
