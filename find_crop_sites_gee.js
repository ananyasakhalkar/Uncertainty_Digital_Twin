// ============================================================
// CROP-SPECIFIC SITE SEARCH
// Wheat / Rice / Soybean
//
// This searches the USDA CDL where available.
// For now, we only use the United States because CDL provides
// crop-specific labels.
//
// Codes:
// 1  = Corn / Maize
// 2  = Cotton
// 3  = Rice
// 5  = Soybeans
// 22 = Durum Wheat
// 23 = Spring Wheat
// 24 = Winter Wheat
// ============================================================

var CDL = ee.Image("USDA/NASS/CDL/2023");

var cropCodes = {
  Rice: [3],
  Soybean: [5],
  Wheat: [22, 23, 24]
};

var regions = {

  Rice: ee.Geometry.Rectangle([
    -95, 25,
    -80, 37
  ]),

  Soybean: ee.Geometry.Rectangle([
    -100, 30,
    -80, 48
  ]),

  Wheat: ee.Geometry.Rectangle([
    -125, 30,
    -95, 50
  ])

};


// ============================================================
// CREATE CROP MASKS
// ============================================================

function makeMask(codes) {

  var mask = CDL.eq(codes[0]);

  for (var i = 1; i < codes.length; i++) {
    mask = mask.or(
      CDL.eq(codes[i])
    );
  }

  return mask;
}


// ============================================================
// SEARCH EACH CROP
// ============================================================

Object.keys(cropCodes).forEach(function(crop) {

  var mask = makeMask(
    cropCodes[crop]
  );

  var region = regions[crop];

  // Sample actual crop pixels.
  var points = mask.selfMask()
    .addBands(CDL)
    .sample({
      region: region,
      scale: 30,
      numPixels: 100,
      geometries: true,
      seed: 42
    });

  print(
    crop + " candidate points",
    points
  );

  // Show on map.
  Map.addLayer(
    mask.selfMask(),
    {},
    crop + " mask"
  );

});

Map.centerObject(
  regions.Wheat,
  4
);
